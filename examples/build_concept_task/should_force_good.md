# Dependency Injection

Your `PaymentService` creates its own `StripeClient` inside the constructor.
Simple, direct, easy to read. Anyone scanning the code can see exactly which
dependency gets used, and there is no extra setup to think about.

That default is attractive because it keeps object creation local and concrete.
