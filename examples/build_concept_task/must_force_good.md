# Dependency Injection

Your `PaymentService` creates a real `StripeClient`, and that choice feels fine
until the test suite runs against production credentials. One test passes, one
fake order turns into a real charge, and now you are explaining to finance why
CI spent money overnight.

After that moment, "just construct the client directly" no longer feels harmless.
