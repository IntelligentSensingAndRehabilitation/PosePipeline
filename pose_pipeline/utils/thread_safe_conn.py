import threading

import datajoint as dj

_dj_thread_local = threading.local()


def make_datajoint_thread_safe() -> None:
    """Patch dj.conn() to return per-thread connections instead of a process-wide singleton.

    DataJoint's default ``dj.conn()`` returns a single pymysql connection shared across
    the entire process.  pymysql is **not** thread-safe — concurrent queries from multiple
    threads corrupt TCP packet sequencing, producing ``Packet sequence number wrong``
    errors.  This affects Streamlit dashboards, ``ThreadPoolExecutor`` workers, FastAPI
    request handlers, and any other threaded context that calls DataJoint.

    After calling this function:

    * The **main thread** continues to use the original singleton connection unchanged.
    * Every **worker thread** gets its own independent ``dj.Connection`` on first use,
      created with the same credentials as the main-thread connection.

    The patch is idempotent — calling it multiple times is safe.

    Usage::

        from pose_pipeline.utils import make_datajoint_thread_safe

        make_datajoint_thread_safe()   # call once at application startup
    """
    if getattr(dj.conn, "_thread_local_patched", False):
        return

    _original_conn = dj.conn

    def _thread_safe_conn(
        host=None,
        user=None,
        password=None,
        *,
        reset=False,
        use_tls=None,
    ):
        if threading.current_thread() is threading.main_thread():
            return _original_conn(host=host, user=user, password=password, reset=reset, use_tls=use_tls)

        if not hasattr(_dj_thread_local, "connection") or reset:
            main = _original_conn()
            h = host if host is not None else main.conn_info["host"]
            port = main.conn_info.get("port")
            u = user if user is not None else main.conn_info["user"]
            p = password if password is not None else main.conn_info["passwd"]
            t = use_tls if use_tls is not None else dj.config.get("database.use_tls")
            _dj_thread_local.connection = dj.Connection(h, u, p, port=port, use_tls=t)

        return _dj_thread_local.connection

    _thread_safe_conn._thread_local_patched = True
    dj.conn = _thread_safe_conn
