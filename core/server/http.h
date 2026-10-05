/* core/server/http.h -- the transport seam.
 *
 * Everything above this line (routing, OpenAI translation, metrics) is written against these
 * plain structs and never sees a socket. That is what lets tests/server_test.cpp drive the real
 * handlers in-process: `Router::dispatch` is the same call the transport makes, minus the port.
 * The alternative -- testing by binding a loopback socket -- tests cpp-httplib, which is already
 * tested, and makes the suite depend on a free port.
 */
#pragma once

#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace rad {
namespace server {

/* Header names are case-insensitive per RFC 9110, and clients disagree about capitalisation of
 * Authorization and Content-Type in particular. */
struct CiLess {
    bool operator()(const std::string& a, const std::string& b) const;
};

struct HttpRequest {
    std::string method = "GET";
    std::string path;                                    /* no query string */
    std::string body;
    std::map<std::string, std::string, CiLess> headers;
    std::map<std::string, std::string> query;
    std::map<std::string, std::string> params;           /* from a ":name" route segment */
    std::string peer;                                    /* "addr:port"; empty in-process */

    /* True once the peer is gone. The transport supplies it; in-process it is absent and the
     * waiting paths simply never see a drop. A dropped connection is a cancellation (spec §14),
     * so every path that waits polls this. */
    std::function<bool()> closed;

    std::string header(const std::string& k, const std::string& def = "") const;
    std::string param(const std::string& k, const std::string& def = "") const;
    bool        peer_gone() const { return closed && closed(); }
};

struct HttpResponse {
    int         status = 200;
    std::string content_type = "application/json; charset=utf-8";
    std::string body;
    std::vector<std::pair<std::string, std::string>> headers;

    /* Streaming. When `next` is set, `body` is ignored and the transport pulls chunks until next()
     * returns false. A chunk may legitimately be empty (nothing produced this poll), in which case
     * nothing is written but the connection is checked. */
    std::function<bool(std::string&)> next;

    /* Runs exactly once when the connection ends, with `ok` false if the peer went away before the
     * stream finished. This is where a dropped stream becomes a scheduler cancellation. */
    std::function<void(bool ok)> on_close;

    bool streaming() const { return next != nullptr; }

    void set_header(std::string k, std::string v) { headers.emplace_back(std::move(k), std::move(v)); }

    static HttpResponse json_body(int status, std::string body);
    static HttpResponse text(int status, std::string body);
};

using Handler = std::function<HttpResponse(const HttpRequest&)>;

/* Exact-match routing plus a single ":name" wildcard segment, which is all the OpenAI surface
 * needs (/v1/models/:id). No regex, no priority rules, nothing to reason about. */
class Router {
public:
    void add(std::string method, std::string path, Handler h);
    void get(std::string path, Handler h)  { add("GET",  std::move(path), std::move(h)); }
    void post(std::string path, Handler h) { add("POST", std::move(path), std::move(h)); }

    /* 404 with a body naming the path, 405 when the path exists under another method. By value,
     * so the transport can move a request -- body included -- into the handler rather than copy
     * a body that may be hundreds of megabytes. */
    HttpResponse dispatch(HttpRequest req) const;

    /* Convenience for tests and for anything that speaks in whole request lines. */
    HttpResponse call(const std::string& method, const std::string& path,
                      const std::string& body = "") const;

    std::vector<std::string> routes() const;

private:
    struct Route {
        std::string method;
        std::vector<std::string> segs;   /* ":name" segments are wildcards */
        Handler h;
    };
    std::vector<Route> routes_;
};

/* Split "/v1/models/foo?x=1" into path and decoded query parameters. */
void parse_target(const std::string& target, std::string* path,
                  std::map<std::string, std::string>* query);
/* Percent-decode. `plus_is_space` is the form encoding, which applies to a query string and
 * not to a path -- see the definition. */
std::string url_decode(const std::string& s, bool plus_is_space = true);

/* ================================================================== the cpp-httplib binding */

/* ONE WORKER PER CONNECTION, for the connection's whole life: httplib reads a request, runs the
 * handler and writes the response on the same thread, and a generation's handler runs until the
 * generation ends. So the pool bounds are connection counts, and Server::run derives them from
 * the admission ceiling rather than from the core count. */
struct TransportOptions {
    std::string host = "0.0.0.0";
    int         port = 8000;              /* 0 -> any free port; see HttpTransport::port */
    int         n_threads = 0;            /* kept alive while idle. 0 -> hardware_concurrency,
                                           * floor 4 */
    int64_t     max_threads = 0;          /* what the pool may grow to. 0 -> n_threads */
    int64_t     max_queued = 0;           /* accepted connections that may wait for a worker once
                                           * it has grown to max_threads; past that they are
                                           * closed. 0 -> unbounded */
    /* The longest the peer may go silent while its REQUEST is being read. It bounds nothing about
     * generation, which is written, not read. A connection that sends part of a request and then
     * goes quiet holds a worker until this runs out, so it is short. */
    int         read_timeout_s = 30;
    int         write_timeout_s = 600;
    int         keep_alive_timeout_s = 5; /* an idle connection's hold on a worker between requests */
    size_t      max_body_bytes = 512ull << 20;   /* images arrive base64 inside the JSON body */
    std::string api_key;                  /* "" disables the check */
    bool        cors = true;
};

class HttpTransport {
public:
    HttpTransport();
    ~HttpTransport();
    HttpTransport(const HttpTransport&) = delete;
    HttpTransport& operator=(const HttpTransport&) = delete;

    /* Blocks until stop(). Returns RAD_OK, or negative if the bind failed. */
    int  serve(const Router& r, const TransportOptions& o);
    void stop();

    /* The port serve() bound, which is the only way to learn it when it was asked for port 0.
     * 0 until the server is accepting connections, so a non-zero answer also means stop() will
     * take effect. */
    int  port() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  /* namespace server */
}  /* namespace rad */
