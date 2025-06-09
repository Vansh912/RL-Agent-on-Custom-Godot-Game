#extends Area2D
#
#@onready var game_manager = %gamemanager
#
#func _on_body_entered(_body: Node2D) -> void:
	#print("+1 coin!")
	#game_manager.add_points()
	#queue_free()
#extends Area2D
#@onready var game_manager = %gamemanager
#var is_active: bool = true
#func _on_body_entered(body: Node2D) -> void:
	#if body.name == "Player":
		#
		#game_manager.add_points(+5)
		#queue_free()
extends Area2D
@onready var game_manager = %gamemanager
@export var value: int = 5
var is_active: bool = true
@onready var original_position: Vector2 = global_position

func _on_body_entered(body: Node2D) -> void:
	if is_active and body.name == "Player":
		game_manager.add_points(value)

		# Let the player track the coin
		if body.has_method("on_coin_collected"):
			body.on_coin_collected(value, self)

		is_active = false
		set_visible(false)
		set_monitoring(false)

func reset_position():
	global_position = original_position
	is_active = true
	set_visible(true)
	set_monitoring(true)
