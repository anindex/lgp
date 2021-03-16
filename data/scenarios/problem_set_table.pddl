(define (problem set_table_prob)
    (:domain set_table)
    (:init
        (agent-free)
        (agent-avoid-human)
        (on plate_blue big_shelf)
    )
    (:goal 
        (and
            (on plate_blue table)
        )
    )
)