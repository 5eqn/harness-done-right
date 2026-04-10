# Connection Pooling

Your API timeout is 200ms. Opening a fresh database connection costs 45ms before
the first query even starts. A single user request triggers four database calls,
and half your latency budget disappears before any useful work happens.

Connection pooling is the technique of keeping a reusable set of live database
connections so requests can borrow one instead of paying that setup cost each time.
