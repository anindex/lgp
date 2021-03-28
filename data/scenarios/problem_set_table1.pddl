(define (problem set_table1)
    (:domain set_table)
    (:objects
        table small_shelf big_shelf - location
        jug - object
    )
    (:init
        (agent-free)
        (agent-avoid-human)
        (on jug big_shelf)
    )
    (:goal 
        (and
            (on jug table)
        )
    )
)