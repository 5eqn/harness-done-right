# Connection Pooling

Connection pooling is a resource-management strategy for database clients. Instead
of opening a brand-new database connection for every request, the application
keeps a reusable pool of already-open connections and lends them to work that
needs one.

An object belongs to connection pooling only if it manages a reusable set of
live connections that can be borrowed and returned across multiple units of work.
A helper that merely stores connection settings, or code that reconnects from
scratch every time, is not connection pooling.
