(define (problem set_table3)
    (:domain set_table)
    (:objects
        table small_shelf big_shelf - location
        cup_red cup_green cup_blue plate_red plate_green plate_blue bowl - object
    )
    (:init
        (agent-free)
        (agent-avoid-human)
        (on cup_green big_shelf)
        (on plate_green big_shelf)
        (on cup_red small_shelf)
        (on plate_red big_shelf)
        (on cup_blue small_shelf)
        (on plate_blue big_shelf)
        (on bowl small_shelf)
    )
    (:goal 
        (and
            (on plate_blue table)
            (agent-at table)
            (on cup_green table)
            (on plate_green table)
            (on cup_red table)
            (on plate_red table)
            (on cup_blue table)
            (on plate_blue table)
            (on bowl table)
        )
    )
)