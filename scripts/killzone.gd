#extends Area2D
#
#@onready var timer: Timer = $Timer
#@onready var score_node = %gamemanager
#var dead = false
#
#func _on_body_entered(body: Node) -> void:
	#if body.name == "Player":
		#print("You Died!")
		#dead = true
		#get_tree().reload_current_scene()

extends Area2D

@onready var game_manager = %gamemanager  # Ensure the node name is capitalized correctly

signal player_died

func _on_body_entered(body: Node) -> void:
	if body.name == "Player":
		game_manager.add_points(-10)  # Apply negative reward
		body.is_dead = true           # Set flag so agent logic can detect terminal state
		body.on_death_zone_entered()  # Trigger reset and episode end
		emit_signal("player_died")
