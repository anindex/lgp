(define (problem set_table1)
    (:domain set_table)
    (:objects
        table small_shelf big_shelf - location
        plate_blue - object
    )
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