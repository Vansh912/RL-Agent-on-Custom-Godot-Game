#extends CharacterBody2D
#
#const SPEED          := 130.0
#const JUMP_VELOCITY  := -300.0
#
#@onready var animated_sprite_2d : AnimatedSprite2D = $AnimatedSprite2D
#var is_dead : bool = false
#
#func _physics_process(delta: float) -> void:
	## gravity & movement
	#if not is_on_floor():
		#velocity.y += ProjectSettings.get_setting("physics/2d/default_gravity") * delta
	#move_and_slide()          # ✅ only here
#
#func apply_action(action: int) -> void:
	## 0=idle, 1=left, 2=right, 3=jump
	#match action:
		#1:
			#velocity.x = -SPEED
			#animated_sprite_2d.flip_h = true
		#2:
			#velocity.x = SPEED
			#animated_sprite_2d.flip_h = false
		#3:
			#if is_on_floor():
				#velocity.y = JUMP_VELOCITY
		#_:
			#velocity.x = 0

#extends Node
#
## --- Hyper-parameters --------------------------------------------------
#const ACTIONS: int = 4                 # 0 = idle, 1 = left, 2 = right, 3 = jump
#var alpha        : float = 0.7        # learning-rate
#var gamma        : float = 0.95        # discount factor
#var epsilon      : float = 0.7        # exploration rate
#var epsilon_decay: float = 0.001       # decay each episode
#var min_epsilon  : float = 0.01
#var episodes     : int   = 0
#
## --- Scene references --------------------------------------------------
#@onready var player       : CharacterBody2D = get_node("/root/Game/Player")
#@onready var game_manager : Node            = %gamemanager      # emits `reward_signal`
#
## --- Q-Table -----------------------------------------------------------
#var q_table: Dictionary = {}            # keys = [state_array, action_id]
#
## ----------------------------------------------------------------------
#func _ready() -> void:
	#game_manager.reward_signal.connect(_on_reward)
	#load_qtable()
#
#func _physics_process(_delta: float) -> void:
	## 1) choose & perform action
	#var state      : Array = _state_hash()
	#var action     : int   = _choose_action(state)
	#player.apply_action(action)
#
	## 2) observe outcome
	#var reward     : int   = game_manager.get_and_reset_reward()
	#var next_state : Array = _state_hash()
	#var done       : bool  = player.is_dead
#
	## 3) learn
	#_update_q(state, action, reward, next_state, done)
#
	## 4) episode wrap-up
	#if done:
		#episodes += 1
		#epsilon = max(min_epsilon, epsilon * epsilon_decay)
		#player.is_dead = false
		#get_tree().reload_current_scene()
		#if episodes % 50 == 0:
			#save_qtable()
#
## ----------------------------------------------------------------------
## ε-greedy with **weighted random exploration**
#func _choose_action(state: Array) -> int:
	#var actions  := [0,1,2, 3]             # action IDs
	#var weights  := [0.01, 0.3,0.3, 0.01]     # bias toward 0 & 1
#
	## explore
	#if randf() < epsilon:
		#return _weighted_random(actions, weights)
#
	## exploit (greedy Q)
	#var best_action := 0
	#var best_q      := -INF
	#for a in actions:
		#var q: float = q_table.get([state, a], 0.0)
		#if q > best_q:
			#best_q      = q
			#best_action = a
	#return best_action
#
## helper: weighted random choice
#func _weighted_random(actions: Array, weights: Array) -> int:
	#var total := 0.0
	#for w in weights:
		#total += w
	#var r := randf() * total
	#var accum := 0.0
	#for i in range(actions.size()):
		#accum += weights[i]
		#if r <= accum:
			#return actions[i]
	#return actions.back()  # fallback (shouldn’t hit)
#
## ----------------------------------------------------------------------
#func _update_q(s: Array, a: int, r: int, s2: Array, done: bool) -> void:
	#var key        := [s, a]
	#var max_next_q := 0.0
	#if not done:
		#for a2 in range(ACTIONS):
			#var q: float = q_table.get([s2, a2], 0.0)
			#if q > max_next_q:
				#max_next_q = q
	#var old_q: float = q_table.get(key, 0.0)
	#var new_q: float = old_q + alpha * (r + gamma * max_next_q - old_q)
	#q_table[key] = new_q
#
## ----------------------------------------------------------------------
#func _state_hash() -> Array:
	## coarse discretisation keeps Q-table small
	#var px: int = int(player.position.x / 32)
	#var py: int = int(player.position.y / 32)
	#var vx: int = int(sign(player.velocity.x))     # -1,0,1
	#var on: int = int(player.is_on_floor())        # 0/1
	#return [px, py, vx, on]
#
## ----------------------------------------------------------------------
#func save_qtable() -> void:
	#var f := FileAccess.open("res://qtable.txt", FileAccess.WRITE)
	#if f:
		#f.store_var(q_table)
		#f.close()
#
#func load_qtable() -> void:
	#if FileAccess.file_exists("res://qtable.txt"):
		#var f := FileAccess.open("res://qtable.txt", FileAccess.READ)
		#if f:
			#q_table = f.get_var()
			#f.close()
#
#func _on_reward(_value: int) -> void:
	#pass    # hook for graphs / debugging
	
extends CharacterBody2D
@onready var Player       : CharacterBody2D = get_node("/root/Game/Player")
@export var speed = 100.00
@export var jump_velocity = -350.0
@export var jump_count = 2
@onready var game_manager : Node = %gamemanager
@export var gravity_multiplier = 2.0

var is_dead: bool = false
var current_jumps = 0
var gravity = ProjectSettings.get_setting("physics/2d/default_gravity")

# --------------------------------------------------------------------------------
# DQN Hyperparameters and training settings
# --------------------------------------------------------------------------------
var q_network : NNET = NNET.new([13, 64, 64, 3], false)
var target_network : NNET = NNET.new([13, 64, 64, 3], false)

var epsilon := 0.8
var gamma := 0.99
var replay_buffer := []
var max_buffer_size := 5000
var batch_size := 64
var target_update_frequency := 1000
var step_count := 0
var train_frequency := 5
var train_counter := 0

# --------------------------------------------------------------------------------
# Tracking environment state
# --------------------------------------------------------------------------------
var previous_distance = 0.0
var previous_state = null
var previous_action = null

var agent_position = Vector2()
var coin_position = Vector2.ZERO

var vision = {}
var episodes = 0


var just_collected_coin = false
var TIME_LIMIT = 60

# Reward logging
var rewards_file_path = "user://episode_rewards.csv"
var current_episode_reward = 0.0
var episode_num = 0
var episode_time = 0.0

var collected_count = 0
var total_collectables = 0
var collectables = []

var time_since_last_coin_collection = 0.0
var initial_position = Vector2()


var jump_cooldown = 0.3  # seconds delay between jumps
var time_since_last_jump = 0.0
# --------------------------------------------------------------------------------
# Initialization
# --------------------------------------------------------------------------------
func _ready() -> void:
	
	stop_training()
	#level = get_parent()
	initial_position = position
	print("position:", position)
	collectables = get_tree().get_nodes_in_group("collectable")
	total_collectables = collectables.size()
	
	# Configure Q-network and target network
	q_network.use_Adam(0.001)
	q_network.set_loss_function(BNNET.LossFunctions.LogCosh_loss)
	q_network.set_function(BNNET.ActivationFunctions.ReLU, 1, 2)
	q_network.set_function(BNNET.ActivationFunctions.identity, 3, 3)
	
	target_network.use_Adam(0.001)
	target_network.set_loss_function(BNNET.LossFunctions.LogCosh_loss)
	target_network.set_function(BNNET.ActivationFunctions.ReLU, 1, 2)
	target_network.set_function(BNNET.ActivationFunctions.identity, 3, 3)
	target_network.assign(q_network)
	
	
	load_agent()

# --------------------------------------------------------------------------------
# Main Training Loop (synchronized every 0.2 seconds)
# --------------------------------------------------------------------------------
var decision_interval := 0.2
var decision_timer := 0.0
var current_action := 0

func _physics_process(delta):
	# Update physics and timers
	apply_gravity(delta)
	time_since_last_coin_collection += delta
	episode_time += delta
	
	agent_position = position
	coin_position = get_nearest_coin_position()
	
	time_since_last_jump += delta
	
	# Execute the current action
	execute_action(current_action, delta)
	
	decision_timer += delta
	if decision_timer >= decision_interval:
		decision_timer = 0.0
		
		# --- Evaluate previous decision ---
		if previous_state != null and previous_action != null:
			var current_state = get_state()
			var current_distance = (agent_position - coin_position).length()
			var reward = get_reward(previous_distance, current_distance, previous_action)
			current_episode_reward += reward
			
			var done = false
			if is_done():
				done = true
				reward -= 10
			
			store_experience(previous_state, previous_action, reward, current_state, done)
			
			previous_state = current_state
			previous_distance = current_distance
			
			if done or collected_count == total_collectables:
				epsilon = max(0.01, epsilon * 0.99)
				print("Episode #", episode_num, " ended. Epsilon:", epsilon)
				print("Episode Reward:", current_episode_reward)
				
				episode_num += 1
				append_episode_reward(episode_num, current_episode_reward, collected_count, epsilon, episode_time)
				
				current_episode_reward = 0
				episode_time = 0
				reset_agent()
				reset_collectables()
				return
			
			train_counter += 1
			step_count += 1
			
			if train_counter % train_frequency == 0 and replay_buffer.size() >= batch_size:
				train_dqn()
			
			if step_count % target_update_frequency == 0:
				target_network.assign(q_network)
		else:
			previous_state = get_state()
			previous_distance = (agent_position - coin_position).length()
		
		# --- Choose a new action ---
		current_action = choose_action(previous_state)
		previous_action = current_action

func is_done() -> bool:
	return time_since_last_coin_collection >= TIME_LIMIT

# --------------------------------------------------------------------------------
# Environment Reset & Control
# --------------------------------------------------------------------------------
func stop_training():
	replay_buffer.clear()
	velocity = Vector2.ZERO
	current_jumps = 0
	time_since_last_coin_collection = 0.0
	previous_state = null
	previous_action = null
	print("Training has been stopped and memory cleared.")

func reset_agent():
	print("reseting agent")
	position = initial_position
	velocity = Vector2.ZERO
	current_jumps = 0
	time_since_last_coin_collection = 0.0
	previous_state = null
	previous_action = null
	print("Episode done. Resetting agent.")

func reset_collectables():
	#var temp_score = level.get_ui_score() * -1
	#level.update_ui_score(temp_score)
	for collectable in collectables:
		collectable.reset_position()
	collected_count = 0

# --------------------------------------------------------------------------------
# State Representation
# --------------------------------------------------------------------------------
func get_state() -> Array:
	var distance_to_coin = coin_position - agent_position
	update_vision()
	
	return [
		distance_to_coin.x / 500,
		distance_to_coin.y / 500,
		velocity.x / 500,
		velocity.y / 500,
		vision.get("RayCast_Up", 1000) / 1000,
		vision.get("RayCast_UpRight", 1000) / 1000,
		vision.get("RayCast_Right", 1000) / 1000,
		vision.get("RayCast_DownRight", 1000) / 1000,
		vision.get("RayCast_Down", 1000) / 1000,
		vision.get("RayCast_DownLeft", 1000) / 1000,
		vision.get("RayCast_Left", 1000) / 1000,
		vision.get("RayCast_UpLeft", 1000) / 1000,
		int(is_on_floor())
	]

func update_vision():
	for child in get_children():
		if child is RayCast2D:
			if child.is_colliding():
				var collision_point = (child.get_collision_point() - global_position).length()
				vision[child.name] = collision_point
			else:
				vision[child.name] = 1000

# --------------------------------------------------------------------------------
# Action Execution
# --------------------------------------------------------------------------------
func execute_action(action: int, delta: float) -> void:
	if is_on_floor():
		current_jumps = 0

	velocity.x = move_toward(velocity.x, 0, speed * delta)

	match action:
		0: # Move right
			velocity.x = speed
		1: # Move left
			velocity.x = -speed
		2: # Jump (or double jump)
			jump_if_possible()

	move_and_slide()

func on_death_zone_entered() -> void:
	if previous_state != null and previous_action != null:
		var current_state = get_state()
		var current_distance = (agent_position - coin_position).length()
		var reward = -10.0  # heavy penalty for falling

		store_experience(previous_state, previous_action, reward, current_state, true)
		
		current_episode_reward += reward
		append_episode_reward(episode_num, current_episode_reward, collected_count, epsilon, episode_time)

		# Reset for next episode
		episode_num += 1
		current_episode_reward = 0.0
		episode_time = 0.0
		reset_agent()
		reset_collectables()
		
		print("Agent fell into kill zone. Episode ended. Reward:", reward)
	else:
		reset_agent()
		reset_collectables()
		print("Agent fell but had no previous state/action.")

func jump_if_possible():
	# Only allow jump if the cooldown period has passed
	if time_since_last_jump < jump_cooldown:
		return
	if current_jumps < jump_count:
		velocity.y = jump_velocity
		current_jumps += 1
		# Reset the cooldown timer
		time_since_last_jump = 0.0

func apply_gravity(delta):
	if not is_on_floor():
		velocity.y += gravity * gravity_multiplier * delta

# --------------------------------------------------------------------------------
# Reward Function (called every decision interval)
# --------------------------------------------------------------------------------
func get_reward(prev_distance: float, current_distance: float, action: int) -> float:
	var reward = 0.0

	if just_collected_coin:
		var coin_bonus = max(0, 100.0 - (time_since_last_coin_collection * 0.5))
		print("Coin bonus: ", coin_bonus)
		time_since_last_coin_collection = 0.0
		just_collected_coin = false
		return coin_bonus

	var dist_improvement = prev_distance - current_distance
	var scaled_improvement = dist_improvement * 0.01
	reward += clamp(scaled_improvement, -0.2, 1.0)

	#var coin_dist = (agent_position - coin_position).length()
	#var bonus = lerp(0.2, 0.0001, clamp(coin_dist / 500.0, 0, 1))
	#reward += bonus
	
	reward -= 0.04
	
	print(reward)
	return reward

# --------------------------------------------------------------------------------
# Find the physically nearest coin (no jump filtering)
# --------------------------------------------------------------------------------
func get_nearest_coin_position() -> Vector2:
	var nearest_coin = null
	var min_dist = INF

	# Get all coins dynamically by group
	var coins = get_tree().get_nodes_in_group("collectable")
	
	for coin in coins:
		var dist = (coin.global_position - agent_position).length()
		if dist < min_dist:
			min_dist = dist
			nearest_coin = coin

	if nearest_coin:
		return nearest_coin.global_position
	else:
		return Vector2()  # Or some default value

# --------------------------------------------------------------------------------
# DQN: Choose Action (Epsilon-Greedy)
# --------------------------------------------------------------------------------
func choose_action(current_state):
	var action = 0
	if randf() < epsilon:
		action = randi() % 3
	else:
		q_network.set_input(current_state)
		q_network.propagate_forward()
		var q_values = q_network.get_output()
		if step_count % 100 == 0:  # Print every 100 steps
			print("Q-values: ", q_values)
		if q_values.size() > 0:
			var max_q = q_values.max()
			var best_actions = []
			for i in range(q_values.size()):
				if q_values[i] == max_q:
					best_actions.append(i)
			if best_actions.size() > 0:
				action = best_actions[randi() % best_actions.size()]
			else:
				action = randi() % 3
		else:
			action = randi() % 3
	return action

# --------------------------------------------------------------------------------
# Replay Buffer Management
# --------------------------------------------------------------------------------
func store_experience(state, action, reward, next_state, done):
	if replay_buffer.size() >= max_buffer_size:
		replay_buffer.pop_front()
	replay_buffer.append([state, action, reward, next_state, done])

func sample_batch():
	var batch = []
	for i in range(batch_size):
		var idx = randi() % replay_buffer.size()
		batch.append(replay_buffer[idx])
	return batch

# --------------------------------------------------------------------------------
# Double DQN Update
# --------------------------------------------------------------------------------
func train_dqn() -> void:
	var batch = sample_batch()
	var batch_states : Array[Array] = []
	var batch_targets : Array[Array] = []

	for experience in batch:
		var state      = experience[0]
		var action     = experience[1]
		var reward     = experience[2]
		var next_state = experience[3]
		var done       = experience[4]

		q_network.set_input(state)
		q_network.propagate_forward()
		var q_values = q_network.get_output()

		q_network.set_input(next_state)
		q_network.propagate_forward()
		var next_q_online = q_network.get_output()

		var best_next_action = 0
		var max_val = -INF
		for i in range(next_q_online.size()):
			if next_q_online[i] > max_val:
				max_val = next_q_online[i]
				best_next_action = i

		target_network.set_input(next_state)
		target_network.propagate_forward()
		var target_output = target_network.get_output()
		var double_q_value = target_output[best_next_action]

		var target_q_value = reward
		if not done:
			target_q_value += gamma * double_q_value

		var target_q_values = q_values.duplicate()
		target_q_values[action] = target_q_value

		batch_states.append(state)
		batch_targets.append(target_q_values)

	q_network.train(batch_states, batch_targets)

# --------------------------------------------------------------------------------
# Handle Collecting a Coin
# --------------------------------------------------------------------------------

#func _on_body_entered(body: CharacterBody2D) -> void:
	#print("player")
	#if body.name == "Player":
		#if body.is_in_group("collectable"):
			#print("got coin")
			#collected_count += 1
			#
			#var coin_value = body.value
			##reward += coin_value  # (if you're using a reward variable for training)
			#print("Collected coin worth ", coin_value, " | Total collected: ", collected_count)		
			#body.collect()
			#just_collected_coin = true
			## Immediately update coin position and previous_distance after collection
			#coin_position = get_nearest_coin_position()
			#previous_distance = (agent_position - coin_position).length()
func on_coin_collected(coin_value: int, coin_node: Node) -> void:
	collected_count += 1
	print("Collected coin worth ", coin_value, " | Total collected: ", collected_count)

	just_collected_coin = true
	coin_position = get_nearest_coin_position()
	print(coin_position)
	previous_distance = (agent_position - coin_position).length()

# --------------------------------------------------------------------------------
# Save / Load Agent Functions
# --------------------------------------------------------------------------------
func save_agent():
	var weights_path = "user://q_network_weights.dat"
	var agent_data_path = "user://agent_data.save"
	var weights_file = FileAccess.open(weights_path, FileAccess.WRITE)
	if weights_file == null:
		push_error("Failed to open weights file for writing: " + weights_path)
	else:
		q_network.save_binary(weights_path)
		print("Q-network weights saved to ", weights_path)
	
	var agent_data = {
		"epsilon": epsilon,
		"gamma": gamma,
		"step_count": step_count,
		"train_counter": train_counter,
	}
	stop_training()
	var agent_file = FileAccess.open(agent_data_path, FileAccess.WRITE)
	if agent_file == null:
		push_error("Failed to open agent data file for writing: " + agent_data_path)
	else:
		agent_file.store_var(agent_data)
		agent_file.close()
		print("Agent parameters saved to ", agent_data_path)

func load_agent():
	print("Agent states reset after loading.")
	var weights_path = "user://q_network_weights.dat"
	var agent_data_path = "user://agent_data.save"
	if not FileAccess.file_exists(weights_path):
		push_error("Weights file does not exist: " + weights_path)
	else:
		q_network.load_data(weights_path)
		print("Q-network weights loaded from ", weights_path)
		target_network.assign(q_network)
		print("Target network synchronized with Q-network.")
	
	if not FileAccess.file_exists(agent_data_path):
		push_error("Agent data file does not exist: " + agent_data_path)
	else:
		var agent_file = FileAccess.open(agent_data_path, FileAccess.READ)
		if agent_file == null:
			push_error("Failed to open agent data file for reading: " + agent_data_path)
		else:
			var agent_data = agent_file.get_var()
			agent_file.close()
			if typeof(agent_data) != TYPE_DICTIONARY:
				push_error("Agent data corrupted or invalid format.")
				return
			epsilon = agent_data.get("epsilon", 1.0)
			gamma = agent_data.get("gamma", 0.9)
			step_count = agent_data.get("step_count", 0)
			train_counter = agent_data.get("train_counter", 0)
			print("Agent parameters loaded from ", agent_data_path)
			epsilon = 0.1
			print("Epsilon set to ", epsilon, " for exploitation.")

func append_episode_reward(episode_number, ep_reward, num_collected, epsilon, ep_time):
	if not FileAccess.file_exists(rewards_file_path):
		var file = FileAccess.open(rewards_file_path, FileAccess.WRITE)
		if file:
			file.store_line("Episode,Reward,Collected,Epsilon,Time")
			file.close()
	
	var file = FileAccess.open(rewards_file_path, FileAccess.READ_WRITE)
	if file:
		file.seek_end()
		file.store_line(
			str(episode_number) + "," +
			str(ep_reward) + "," +
			str(num_collected) + "," +
			str(epsilon) + "," +
			str(ep_time)
		)
		file.close()
