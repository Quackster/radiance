#include "http.h"
#include "rad_internal.h"

/* httplib pulls in <thread>, <sys/socket.h> and a great deal else. It is included in exactly this
 * translation unit so that nothing above the transport seam pays for it. */
#include "cpp-httplib/httplib.h"

#include <algorithm>
#include <atomic>
#include <cctype>
#include <chrono>
#include <condition_variable>
#include <list>
#include <mutex>
#include <system_error>
#include <thread>
#include <vector>

namespace rad {
namespace server {

bool CiLess::operator()(const std::string& a, const std::string& b) const {
    return std::lexicographical_compare(
        a.begin(), a.end(), b.begin(), b.end(),
        [](unsigned char x, unsigned char y) { return std::tolower(x) < std::tolower(y); });
}

std::string HttpRequest::header(const std::string& k, const std::string& def) const {
    auto it = headers.find(k);
    return it == headers.end() ? def : it->second;
}

std::string HttpRequest::param(const std::string& k, const std::string& def) const {
    auto it = params.find(k);
    if (it != params.end()) return it->second;
    auto q = query.find(k);
    return q == query.end() ? def : q->second;
}

HttpResponse HttpResponse::json_body(int status, std::string body) {
    HttpResponse r;
    r.status = status;
    r.body = std::move(body);
    return r;
}

HttpResponse HttpResponse::text(int status, std::string body) {
    HttpResponse r;
    r.status = status;
    r.content_type = "text/plain; charset=utf-8";
    r.body = std::move(body);
    return r;
}

/* ------------------------------------------------------------------ target parsing */

/* `+` IS A SPACE IN A QUERY STRING AND A PLUS SIGN IN A PATH. The two halves of a target are
 * decoded by different rules: the form encoding that gives `+` its second meaning applies to
 * `application/x-www-form-urlencoded` data, which is what a query string is and what a path is
 * not. Decoding a path the query's way renames a route -- /v1/models/gpt+4 becomes a request for
 * "gpt 4", which resolves to nothing and reports the name it was not asked about. */
std::string url_decode(const std::string& s, bool plus_is_space) {
    std::string out;
    out.reserve(s.size());
    auto hex = [](char c) -> int {
        if (c >= '0' && c <= '9') return c - '0';
        if (c >= 'a' && c <= 'f') return c - 'a' + 10;
        if (c >= 'A' && c <= 'F') return c - 'A' + 10;
        return -1;
    };
    for (size_t i = 0; i < s.size(); ++i) {
        if (s[i] == '%' && i + 2 < s.size()) {
            int h = hex(s[i + 1]), l = hex(s[i + 2]);
            if (h >= 0 && l >= 0) { out.push_back((char)(h * 16 + l)); i += 2; continue; }
        }
        out.push_back(plus_is_space && s[i] == '+' ? ' ' : s[i]);
    }
    return out;
}

void parse_target(const std::string& target, std::string* path,
                  std::map<std::string, std::string>* query) {
    size_t q = target.find('?');
    if (path) *path = url_decode(target.substr(0, q), /*plus_is_space=*/false);
    if (!query || q == std::string::npos) return;
    size_t i = q + 1;
    while (i < target.size()) {
        size_t amp = target.find('&', i);
        if (amp == std::string::npos) amp = target.size();
        std::string kv = target.substr(i, amp - i);
        size_t eq = kv.find('=');
        if (eq == std::string::npos) (*query)[url_decode(kv)] = "";
        else (*query)[url_decode(kv.substr(0, eq))] = url_decode(kv.substr(eq + 1));
        i = amp + 1;
    }
}

/* ------------------------------------------------------------------ routing */

static std::vector<std::string> split_path(const std::string& p) {
    std::vector<std::string> out;
    size_t i = 0;
    while (i < p.size()) {
        while (i < p.size() && p[i] == '/') ++i;
        size_t j = i;
        while (j < p.size() && p[j] != '/') ++j;
        if (j > i) out.push_back(p.substr(i, j - i));
        i = j;
    }
    return out;
}

void Router::add(std::string method, std::string path, Handler h) {
    routes_.push_back(Route{std::move(method), split_path(path), std::move(h)});
}

std::vector<std::string> Router::routes() const {
    std::vector<std::string> out;
    for (auto& r : routes_) {
        std::string p;
        for (auto& s : r.segs) { p += '/'; p += s; }
        out.push_back(r.method + " " + (p.empty() ? "/" : p));
    }
    return out;
}

static std::string error_body(const char* type, const std::string& msg) {
    /* Hand-rolled rather than routed through nlohmann, because this is the one error path that
     * must work even if the JSON layer is what failed. */
    std::string esc;
    for (char c : msg) {
        if (c == '"' || c == '\\') { esc.push_back('\\'); esc.push_back(c); }
        else if (c == '\n') esc += "\\n";
        else if ((unsigned char)c < 0x20) esc += "?";
        else esc.push_back(c);
    }
    return std::string("{\"error\":{\"message\":\"") + esc + "\",\"type\":\"" + type +
           "\",\"param\":null,\"code\":null}}";
}

HttpResponse Router::dispatch(HttpRequest req) const {
    auto segs = split_path(req.path);
    bool path_matched = false;

    for (const auto& r : routes_) {
        if (r.segs.size() != segs.size()) continue;
        std::map<std::string, std::string> caps;
        bool ok = true;
        for (size_t i = 0; i < segs.size(); ++i) {
            if (!r.segs[i].empty() && r.segs[i][0] == ':') caps[r.segs[i].substr(1)] = segs[i];
            else if (r.segs[i] != segs[i]) { ok = false; break; }
        }
        if (!ok) continue;
        path_matched = true;
        if (r.method != req.method) continue;

        req.params.insert(caps.begin(), caps.end());
        return r.h(req);
    }

    if (path_matched)
        return HttpResponse::json_body(
            405, error_body("invalid_request_error",
                            req.method + " is not allowed on " + req.path));
    return HttpResponse::json_body(
        404, error_body("not_found_error", "no route for " + req.method + " " + req.path));
}

HttpResponse Router::call(const std::string& method, const std::string& path,
                          const std::string& body) const {
    HttpRequest r;
    r.method = method;
    parse_target(path, &r.path, &r.query);
    r.body = body;
    if (!body.empty()) r.headers["Content-Type"] = "application/json";
    return dispatch(std::move(r));
}

/* ------------------------------------------------------------------ the worker pool */

/* A WORKER FOR EVERY CONNECTION, UP TO A BOUND, STARTED THE MOMENT THE CONNECTION HAS NONE.
 *
 * httplib's own ThreadPool starts an extra worker only when it counts no idle one, and a worker
 * counts as idle from the moment it is woken until it has taken its job. Two connections arriving
 * inside that window leave the second queued with no worker started for it -- and a generation
 * holds its worker for minutes, so that connection waits for the next one to arrive or for a
 * generation to end. This pool compares the queue with the number of workers free to take from
 * it, so a connection waits only once the pool is at its bound.
 *
 * `keep` workers stay up; the rest exit after kIdleExit with nothing to do. Past `most` workers,
 * `queued` connections may wait (0: any number) and further ones are refused, which httplib
 * answers by closing the socket. */
class ConnectionPool final : public httplib::TaskQueue {
public:
    ConnectionPool(size_t keep, size_t most, size_t queued)
        : keep_(keep), most_(std::max(most, keep)), queued_(queued) {
        std::lock_guard<std::mutex> lk(m_);
        for (size_t i = 0; i < keep_; ++i) start();
    }
    ~ConnectionPool() override { shutdown(); }

    bool enqueue(std::function<void()> fn) override {
        {
            std::lock_guard<std::mutex> lk(m_);
            if (stop_) return false;
            if (jobs_.size() >= free_) {       /* no free worker is left for this job */
                const bool started = threads_.size() < most_ && start();
                if (!started && queued_ && jobs_.size() - free_ >= queued_) return false;
            }
            jobs_.push_back(std::move(fn));
        }
        cv_.notify_one();
        return true;
    }

    /* Runs what is queued, then joins every worker. Idempotent: httplib calls it and then
     * destroys the pool, which calls it again. */
    void shutdown() override {
        std::list<std::thread> live;
        {
            std::lock_guard<std::mutex> lk(m_);
            stop_ = true;
            live.swap(threads_);
        }
        cv_.notify_all();
        for (auto& t : live) t.join();
        std::lock_guard<std::mutex> lk(m_);
        reap();
    }

private:
    static constexpr std::chrono::seconds kIdleExit{5};

    /* With m_ held. False when the system will not give us a thread; the job then waits. */
    bool start() {
        reap();
        try {
            threads_.emplace_back([this] { work(); });
        } catch (const std::system_error& e) {
            RAD_WARN("server: cannot start an HTTP worker (%s); the connection waits", e.what());
            return false;
        }
        ++free_;
        return true;
    }

    /* With m_ held: join the workers that exited idle. They left m_ before exiting, so this
     * waits only for their last instructions. */
    void reap() {
        for (auto& t : finished_) t.join();
        finished_.clear();
    }

    void work() {
        std::unique_lock<std::mutex> lk(m_);
        for (;;) {
            while (jobs_.empty() && !stop_) {
                if (threads_.size() <= keep_) { cv_.wait(lk); continue; }
                if (cv_.wait_for(lk, kIdleExit) == std::cv_status::timeout && jobs_.empty() &&
                    !stop_ && threads_.size() > keep_) {
                    /* Hand this thread's handle to finished_, for the next start() to join. */
                    for (auto it = threads_.begin(); it != threads_.end(); ++it) {
                        if (it->get_id() != std::this_thread::get_id()) continue;
                        finished_.push_back(std::move(*it));
                        threads_.erase(it);
                        break;
                    }
                    --free_;
                    return;
                }
            }
            if (jobs_.empty()) { --free_; return; }    /* stopping, and nothing is left to run */
            std::function<void()> fn = std::move(jobs_.front());
            jobs_.pop_front();
            --free_;
            lk.unlock();
            fn();
            lk.lock();
            ++free_;
        }
    }

    const size_t keep_, most_, queued_;
    std::mutex m_;
    std::condition_variable cv_;
    std::list<std::function<void()>> jobs_;
    std::list<std::thread> threads_;       /* every worker not yet exited */
    std::vector<std::thread> finished_;    /* exited idle, not yet joined */
    size_t free_ = 0;                      /* workers not running a job */
    bool stop_ = false;
};

/* ------------------------------------------------------------------ the httplib binding */

struct HttpTransport::Impl {
    httplib::Server  svr;
    std::atomic<int> port{0};
};

HttpTransport::HttpTransport() : impl_(new Impl) {}
HttpTransport::~HttpTransport() { stop(); }

void HttpTransport::stop() { if (impl_) impl_->svr.stop(); }

/* Not before the accept loop is running: until then httplib's stop() has nothing to stop and
 * does nothing, so a caller that took the port as "up" and then stopped would wait forever. */
int HttpTransport::port() const {
    return impl_ && impl_->svr.is_running() ? impl_->port.load() : 0;
}

static void apply(const HttpResponse& r, httplib::Response& res) {
    res.status = r.status;
    for (auto& h : r.headers) res.set_header(h.first, h.second);
}

int HttpTransport::serve(const Router& router, const TransportOptions& o) {
    auto& svr = impl_->svr;

    /* A REQUEST LINE LONGER THAN httplib's LINE LIMIT (CPPHTTPLIB_MAX_LINE_LENGTH) GETS NO STATUS.
     * Its line reader gives up before the request exists, and no handler, error handler or logger
     * runs, so there is nothing here that could answer 414: the connection is closed unanswered.
     * With the build's CPPHTTPLIB_REQUEST_URI_MAX_LENGTH equal to that limit, httplib's own 414
     * cannot fire either, because a target that long never fits in a line that is read. */
    svr.set_read_timeout(o.read_timeout_s, 0);
    svr.set_write_timeout(o.write_timeout_s, 0);
    svr.set_keep_alive_timeout(o.keep_alive_timeout_s);
    svr.set_payload_max_length(o.max_body_bytes);

    int nt = o.n_threads > 0 ? o.n_threads : (int)std::thread::hardware_concurrency();
    if (nt < 4) nt = 4;
    const size_t most = (size_t)std::max<int64_t>(o.max_threads, nt);
    const size_t queued = (size_t)std::max<int64_t>(o.max_queued, 0);
    svr.new_task_queue = [nt, most, queued] { return new ConnectionPool((size_t)nt, most, queued); };

    /* A handler must never throw across the transport: an exception here closes the connection
     * with no body at all, which is the least diagnosable failure a client can get. */
    svr.set_exception_handler([](const httplib::Request&, httplib::Response& res,
                                 std::exception_ptr ep) {
        std::string what = "unhandled exception in handler";
        try { std::rethrow_exception(ep); }
        catch (const std::exception& e) { what = e.what(); }
        catch (...) {}
        RAD_ERR("server: %s", what.c_str());
        res.status = 500;
        res.set_content(error_body("internal_server_error", what), "application/json");
    });

    const std::string api_key = o.api_key;
    const bool cors = o.cors;

    /* What a request must pass before its body is read, so a refused one never has it buffered. */
    auto gate = [api_key, cors](const httplib::Request& hreq, httplib::Response& hres) -> bool {
        if (cors) {
            hres.set_header("Access-Control-Allow-Origin", "*");
            hres.set_header("Access-Control-Allow-Headers", "Content-Type, Authorization");
            hres.set_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS");
        }
        /* THE PROBE PATHS ARE EXEMPT, AND THE DASHBOARD PAGE. A load balancer cannot carry a
         * bearer token, so an authenticated /health is a server that reports itself down. The
         * page is the one static document the server has: it holds no data, and a browser opening
         * it cannot send a header, so it loads and then sends the key on every fetch of data it
         * makes. Everything else is guarded. vLLM's middleware guards only paths under /v1,
         * leaving /tokenize open, which hands an unauthenticated caller the vocabulary and the
         * chat template. This does not. */
        const std::string& p = hreq.path;
        if (!api_key.empty() && p != "/health" && p != "/ping" && p != "/" && p != "/dashboard") {
            std::string got = hreq.get_header_value("Authorization");
            const std::string want = "Bearer " + api_key;
            if (got != want) {
                hres.status = 401;
                hres.set_content(error_body("authentication_error", "invalid api key"),
                                 "application/json");
                return false;
            }
        }
        return true;
    };

    auto bridge = [&router](const httplib::Request& hreq, std::string body,
                            httplib::Response& hres) {
        HttpRequest req;
        req.method = hreq.method;
        req.path = hreq.path;
        req.body = std::move(body);
        req.peer = hreq.remote_addr + ":" + std::to_string(hreq.remote_port);
        for (auto& h : hreq.headers) req.headers[h.first] = h.second;
        for (auto& p : hreq.params) req.query[p.first] = p.second;
        req.closed = [&hreq] { return hreq.is_connection_closed(); };

        HttpResponse res = router.dispatch(std::move(req));
        apply(res, hres);

        if (!res.streaming()) {
            hres.set_content(std::move(res.body), res.content_type);
            if (res.on_close) res.on_close(true);
            return;
        }

        /* Streaming. The provider owns the handler's state via the shared_ptr, because httplib
         * keeps it alive past this function; `releaser` is what tells us whether the peer went
         * away, and it is the hook cancellation hangs off. */
        auto st = std::make_shared<HttpResponse>(std::move(res));
        hres.set_chunked_content_provider(
            st->content_type,
            [st](size_t /*offset*/, httplib::DataSink& sink) -> bool {
                std::string chunk;
                bool more = st->next(chunk);
                if (!chunk.empty()) {
                    if (!sink.write(chunk.data(), chunk.size())) return false;  /* peer gone */
                }
                if (!more) sink.done();
                return more;
            },
            [st](bool ok) { if (st->on_close) st->on_close(ok); });
    };

    for (const auto& sig : router.routes()) {
        size_t sp = sig.find(' ');
        std::string method = sig.substr(0, sp);
        std::string path = sig.substr(sp + 1);
        /* ":name" is httplib's own wildcard spelling too, so the pattern carries across. */
        if (method == "GET") {
            svr.Get(path, [gate, bridge](const httplib::Request& q, httplib::Response& s) {
                if (gate(q, s)) bridge(q, q.body, s);
            });
            continue;
        }
        /* A POST BODY IS READ STRAIGHT INTO THE STRING THE HANDLER PARSES. Left to httplib it
         * lands in its own Request first, which only hands it out const, so it would be copied
         * into ours: a body at the size limit held twice before a byte of it is looked at. A
         * multipart body is not read at all -- no endpoint takes one, and httplib discards what a
         * handler leaves unread -- so the handler sees an empty body and refuses it. */
        svr.Post(path, [gate, bridge](const httplib::Request& q, httplib::Response& s,
                                      const httplib::ContentReader& read) {
            if (!gate(q, s)) return;
            std::string body;
            if (!q.is_multipart_form_data() &&
                !read([&body](const char* p, size_t n) { body.append(p, n); return true; })) {
                /* httplib has already set 413 for a body over the limit; anything else is a peer
                 * that stopped sending partway, which has to be answered as something. */
                if (s.status < 0) {
                    s.status = 400;
                    s.set_content(error_body("invalid_request_error",
                                             "the request body could not be read"),
                                  "application/json");
                }
                return;
            }
            bridge(q, std::move(body), s);
        });
    }
    svr.Options(".*", [cors](const httplib::Request&, httplib::Response& res) {
        if (cors) {
            res.set_header("Access-Control-Allow-Origin", "*");
            res.set_header("Access-Control-Allow-Headers", "Content-Type, Authorization");
            res.set_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS");
        }
        res.status = 204;
    });

    /* Bound once: asking for a named port and then for any port would leave the first socket
     * bound and unused. */
    const int port = o.port == 0 ? svr.bind_to_any_port(o.host.c_str())
                     : svr.bind_to_port(o.host.c_str(), o.port) ? o.port : -1;
    if (port <= 0) {
        RAD_ERR("server: cannot bind %s:%d", o.host.c_str(), o.port);
        return RAD_E_IO;
    }
    impl_->port.store(port);
    RAD_INFO("server: listening on http://%s:%d (%d threads, up to %zu)", o.host.c_str(), port,
             nt, most);
    return svr.listen_after_bind() ? RAD_OK : RAD_E_IO;
}

}  /* namespace server */
}  /* namespace rad */
