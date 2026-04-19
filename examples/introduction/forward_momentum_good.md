# Circuit Breaker

Service A calls Service B. B starts hanging for 3 seconds at a time. Soon A is
full of waiting threads, then Service C starts timing out because it depends on
A, and the slowdown spreads one hop at a time.

You do not just want a definition here. You want to know how to stop one sick
dependency from infecting everything upstream.
