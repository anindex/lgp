(define (domain set_table)
  (:requirements :typing)
  (:types location object)
  (:constants
    table small_shelf big_shelf - location
    cup_red cup_green cup_pink plate_blue plate_green plate_red plate_pink bowl jug - object
  )
  (:predicates
    (agent-at ?l - location)
    (human-at ?l - location)
    (on ?x - object ?l - location)
    (agent-free)
    (agent-avoid-human)
    (agent-carry ?x - object)
    (human-carry ?x - object)
  )
  (:functions (move-time ?l - location))

  (:durative-action move
      :parameters (?a - agent ?l - location)
      :duration (= ?duration (move-time ?l))
      :precondition (at end (not (human-at ?l)))
      :effect (and (at end (not (agent-at ?*))) (at end (at ?a ?l)))
  )
  (:durative-action pick
      :parameters (?a - agent ?x - object ?l - location)
      :duration (= ?duration 2)
      :precondition (and (at start (agent-at ?l)) (at start (on ?x ?l)) (at start (agent-free))) 
      :effect (and (at end (not (on ?x ?l))) (at end (not (agent-free))) (at end (agent-carry ?x)))
  )
  (:durative-action place
      :parameters (?a - agent ?x - object ?l - location)
      :duration (= ?duration 2)
      :precondition (and (at start (agent-at ?l)) (at start (agent-carry ?x)))  
      :effect (and (at end (not (agent-carry ?x))) (at end (on ?x ?l)) (at end (agent-free)))
  )
)