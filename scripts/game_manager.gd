#extends Node
#
#var score = 0
#var reward : int = 0  
#
#func add_points(amount: int = 1) -> void:
	#score  += amount
	#reward  = amount  
	#
		#
#func get_score() -> int:
	#return score
#
#func get_and_reset_reward() -> int:
	#var r := reward
	#reward = 0             # reset so it’s only returned once
	#return r

extends Node
signal reward_signal(value)

var score  : int = 0
var reward : int = 0          # last reward

func add_points(amount: int) -> void:
	score  += amount
	reward  += amount          # accumulate if multiple coins this frame
	emit_signal("reward_signal", amount)

func get_and_reset_reward() -> int:
	var r := reward
	reward = 0
	return r
func reset_environment() -> void:
	# Reset player
	var player = get_tree().get_first_node_in_group("player")
	if player:
		player.reset()

	# Reset all coins
	for coin in get_tree().get_nodes_in_group("collectable"):
		if coin.has_method("reset_position"):
			coin.reset_position()
