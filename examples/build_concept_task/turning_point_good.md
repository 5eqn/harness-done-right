# Connection Pooling

Each request keeps paying 45ms to create a new database connection, and the math
gets worse as traffic rises. Instead of opening a fresh connection every time,
keep a small set of connections alive, borrow one for the query, then return it
to the pool when the work finishes.

Connection pooling is that move: reusing a managed pool of live connections so
the setup cost happens rarely instead of once per operation.
