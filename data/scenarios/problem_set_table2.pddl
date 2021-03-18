(define (problem set_table2)
    (:domain set_table)
    (:objects
        table small_shelf big_shelf - location
        cup_green plate_blue - object
    )
    (:init
        (agent-free)
        (agent-avoid-human)
        (on cup_green small_shelf)
        (on plate_blue big_shelf)
    )
    (:goal 
        (and
            (on plate_blue table)
            (on cup_green table)
            (agent-at table)
        )
    )
)