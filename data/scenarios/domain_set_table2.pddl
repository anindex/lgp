(define (domain set_table)
  (:requirements :typing)
  (:types location object)
  (:constants
    table small_shelf big_shelf - location
    cup_red cup_green cup_blue cup_pink plate_pink plate_red plate_green plate_blue jug bowl - object
  )
  (:predicates
    (agent-at ?l - location)
    (on ?x - object ?l - location)
    (agent-free)
    (agent-avoid-human)
    (agent-carry ?x - object)
    (human-carry ?x - object)
  )

  (:action move
      :parameters (?l - location)
      :precondition ()
      :effect (and (not (agent-at ?*)) (agent-at ?l))
  )
  (:action pick
      :parameters (?x - object ?l - location)
      :precondition (and (agent-at ?l) (on ?x ?l) (not (human-carry ?x)) (at start (agent-free))) 
      :effect (and (not (on ?x ?l)) (not (agent-free)) (agent-carry ?x))
  )
  (:action place
      :parameters (?x - object ?l - location)
      :precondition (and (agent-at ?l) (agent-carry ?x)) 
      :effect (and (not (agent-carry ?x)) (on ?x ?l) (agent-free))
  )
)