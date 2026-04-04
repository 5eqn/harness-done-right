"""
Task specification - THIS FILE IS IMMUTABLE ONCE AGREED
Defines the formal requirements for the task
"""

from hdr import File, PythonWorkspace, Concept, Task
from pydantic import Field


class IntroSection(Task):
    """
    Introduction section that explains what the subject is and why it matters.
    """

    concept: Concept = Field(
        description="Authoritative definition of the concept being introduced"
    )
    file: File = Field(
        description="Markdown file (.md) containing the introduction text"
    )

    def __init__(self, **data):
        super().__init__(**data)

        # Basic correctness verifications
        basic_verifications = [
            "The file does not contradict any statement in the concept.",
            "The file only discusses topics within the scope indicated by concept name; it does not digress into unrelated subjects.",
            "Every technical term used in file is either defined in the concept or explained within file itself.",
        ]

        # Story quality verifications
        story_verifications = [
            # Verification 1: Opening - Concrete Scenario (predicates on first 2-3 sentences)
            """
            The first 2-3 sentences present a concrete scenario with specific details 
            (numbers, actions, observable outcomes) that a reader familiar with the context 
            can immediately visualize, rather than abstract definitions or generic claims.
            
            PASSES:
            - Concept: Connection Pooling
              Opening: "Your API timeout is 200ms. A database connection takes 50ms to establish. 
              Four services deep in a call chain, and you've already failed before any query runs."
              Why: Specific numbers (200ms, 50ms, four services), observable outcome (failed before query runs).
            
            - Concept: Cache Invalidation
              Opening: "You deploy on Friday evening. Monday morning, users report seeing last week's prices. 
              Your cache is serving stale data, and every manual fix you attempt breaks something else."
              Why: Specific time (Friday, Monday), specific symptom (last week's prices), tangible struggle (manual fix breaks things).
            
            FAILS:
            - Concept: Connection Pooling
              Opening: "Connection pooling is a technique for managing database connections efficiently, 
              improving performance and resource utilization in modern applications."
              Why: No scenario, no numbers, no observable event—pure abstraction.
            
            - Concept: Cache Invalidation
              Opening: "Caching is important for modern applications because it improves performance 
              and reduces load on backend systems, though it introduces complexity."
              Why: Generic claim ("important", "complexity"), no specific situation the reader can see.
            """,

            # Verification 2: Opening - Forward Momentum (predicates on first 2-3 sentences)
            """
            After reading the first 2-3 sentences, a reader would want to continue not merely 
            "to learn about the topic" but to witness how a specific tension unfolds or resolves—
            they can articulate a concrete "what happens next" or "how is this resolved" question.
            
            PASSES:
            - Concept: Database Indexing
              Opening: "Your query runs in 2ms with 10,000 rows. The table grows to 10 million. 
              The same query now takes 8 seconds—your users see a loading spinner, then leave."
              Why: Reader asks "How do I make it fast again at 10 million rows?"—a concrete resolution question.
            
            - Concept: Circuit Breaker Pattern
              Opening: "Service A calls Service B. B is slow—3 second timeout. A waits. Now A is slow. 
              Service C calls A. Now C is slow too. One sick service infects your entire system."
              Why: Reader asks "How do I stop the infection?"—they want to see the intervention.
            
            FAILS:
            - Concept: Database Indexing
              Opening: "Database indexing is a technique to speed up data retrieval operations 
              by creating efficient lookup structures."
              Why: Reader might ask "What is an index?" but not "What happens next?"—no tension to resolve.
            
            - Concept: Circuit Breaker Pattern
              Opening: "The circuit breaker pattern is a design pattern used in distributed systems 
              to prevent cascading failures and improve resilience."
              Why: Reader has no event to follow, no conflict to witness unfold.
            """,

            # Verification 3: Contains "Should" Force
            """
            The file contains a vivid depiction of the "should" force: the natural expectation, 
            default behavior, or inertia that exists before or without the concept—what the world 
            assumes, how things normally work, or the path of least resistance that seems sufficient.
            
            PASSES:
            - Concept: Dependency Injection
              Text contains: "Your PaymentService creates its own StripeClient inside the constructor. 
              Simple, direct, works. The code reads exactly like what it does."
              Why: Shows the natural default (create your own dependency), its appeal (simple, direct), 
              the inertia (why would you change what works?).
            
            - Concept: Immutability
              Text contains: "You update the user's balance in place. One line of code. 
              The object changes, your function returns, done."
              Why: Shows the default (mutate in place), its appeal (one line, done), 
              the assumed sufficiency (why complicate it?).
            
            FAILS:
            - Concept: Dependency Injection
              Text contains: "Without dependency injection, code becomes tightly coupled 
              and difficult to test."
              Why: States a problem but doesn't show the *appeal* of the default—
              the reader doesn't feel why someone would do it the "wrong" way.
            
            - Concept: Immutability
              Text contains: "Mutable state can lead to bugs in concurrent programs."
              Why: Jumps to problems without showing why mutation feels natural, easy, and tempting first.
            """,

            # Verification 4: Contains "Must" Force (Prevailing)
            """
            The file contains a vivid depiction of the "must" force that prevails over the "should" force:
            a specific moment where the default breaks down undeniably, and the reader—while still 
            feeling the appeal of the "should" force—cannot unsee the necessity of change. 
            The "must" force should be the more memorable and compelling of the two.
            
            PASSES:
            - Concept: Dependency Injection
              Text contains: "You run the test. It passes. It also charged a real credit card. 
              Your StripeClient was real, and now you're explaining to finance why the test suite 
              costs $847 per run."
              Why: The "should" force (direct instantiation) still feels simple, but this moment 
              is undeniable—you cannot argue with $847 and a real charge. The "must" force wins.
            
            - Concept: Immutability
              Text contains: "Three functions hold a reference to the same user object. 
              One of them changes the balance. You spend four hours tracing which function, 
              on which code path, in what order. You find it. Next week, it happens again."
              Why: The "should" force (mutation is easy) still appeals, but four hours twice 
              is a cost the reader cannot dismiss. The "must" force becomes irresistible.
            
            FAILS:
            - Concept: Dependency Injection
              Text contains: "Creating dependencies directly makes unit testing impossible."
              Why: "Impossible" is stated but not shown—reader can still imagine it's fine for their case.
              The "must" force is asserted, not demonstrated.
            
            - Concept: Immutability
              Text contains: "Shared mutable state is a common source of bugs."
              Why: "Common source" is statistics, not a moment. The reader doesn't witness failure,
              so the "should" force (mutation is easy) still feels stronger.
            """,

            # Verification 5: Contains Clear Turning Point
            """
            The file contains a clear turning point where the concept is introduced as the response 
            to the tension between "should" and "must" forces—not as a feature list or definition, 
            but as the move that resolves or transforms the situation the reader has just witnessed.
            
            PASSES:
            - Concept: Dependency Injection
              Text contains: "What if PaymentService didn't create its StripeClient? 
              What if it received one—and didn't care whether it was real or fake?"
              Why: The concept enters as a question that directly answers the witnessed problem.
              The reader sees it as a move in the story, not a definition.
            
            - Concept: Connection Pooling
              Text contains: "Instead of opening a connection for each request, you keep a set 
              of connections alive. Borrow one, use it, return it. The 50ms cost happens once, 
              not per request."
              Why: The concept is framed as the action that breaks the arithmetic trap shown earlier.
              The reader sees cause and effect, not a feature.
            
            FAILS:
            - Concept: Dependency Injection
              Text contains: "Dependency injection is a design pattern where dependencies are 
              provided to a class rather than created internally."
              Why: Pure definition. No connection to the tension. Reader must do the work 
              of linking this to the story themselves.
            
            - Concept: Connection Pooling
              Text contains: "Connection pooling provides better resource utilization 
              and reduces connection overhead."
              Why: Benefit list, not a turning point. Doesn't show how this resolves 
              the specific situation that was set up.
            """,
        ]

        for v in basic_verifications:
            self.verify(v)

        for v in story_verifications:
            self.verify(v)


class UsageSection(Task):
    """
    Usage section that explains how to use a concept with runnable examples.
    """

    concept: Concept = Field(
        description="Authoritative definition of the concept being documented"
    )
    file: File = Field(
        description="Markdown file (.md) containing the usage explanation"
    )
    code_examples: PythonWorkspace = Field(
        description="Python workspace containing runnable code examples"
    )

    def __init__(self, **data):
        super().__init__(**data)

        self.verify("The file does not contradict any statement in the concept.")
        self.verify(
            "Every code snippet in file has a corresponding file in code_examples, and every file in code_examples is referenced in file."
        )
        self.verify(
            "The file only explains usage of concept; it does not introduce or explain usage of unrelated concepts."
        )
        self.verify(
            "The file contains no time-sensitive terms (e.g., 'currently', 'recently', 'as of now') without specifying an exact version or date."
        )
        self.verify(
            "Every technical term used in file is either defined in the concept or explained within file itself."
        )


class Documentation(Task):
    """
    Complete documentation combining introduction and usage sections.
    """

    intro: IntroSection = Field(
        description="Introduction section explaining what and why"
    )
    usage: UsageSection = Field(
        description="Usage section explaining how with examples"
    )
