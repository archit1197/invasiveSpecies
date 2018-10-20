from constants import *
import math
import numpy as np
from pulp import *
from util import UpperP

def ddvupper(mdp, start_state=0, epsilon=4, delta=0.1):

	global Vmax
	print Vmax
	initial_iterations = 10*mdp.numStates*mdp.numActions
	### Estimate the horizon based on Fiechter
	H = int((math.log(Vmax) + math.log(6.0/epsilon))/(1-mdp.discountFactor))
	c = 2
	it=0
	samples = 0

	### Calculating m based on the parameters

	first_term = mdp.numStates/(epsilon**2*(1-mdp.discountFactor)**4)
	second_term = math.log(mdp.numStates*mdp.numActions/(epsilon*(1-mdp.discountFactor)*delta))/(epsilon**2*(1-mdp.discountFactor)**4)
	m = c*(first_term+second_term)
	delta = delta/(mdp.numStates*mdp.numActions*m)
	print "Chosen value of m is :", m
	N_s_a = np.array([[0 for i in range(mdp.numActions)] for j in range(mdp.numStates)])
	N_s_a_sprime = np.array([[[0 for i in range(mdp.numStates)] for j in range(mdp.numActions)] for k in range(mdp.numStates)])
	P_s_a_sprime = np.array([[[0.0 for i in range(mdp.numStates)] for j in range(mdp.numActions)] for k in range(mdp.numStates)])
	P_tilda = [[[0.0 for i in range(mdp.numStates)] for j in range(mdp.numActions)] for k in range(mdp.numStates)]
	R_s_a = [[0.0 for i in range(mdp.numActions)] for j in range(mdp.numStates)]
	Qupper = [[Vmax for i in range(mdp.numActions)] for j in range(mdp.numStates)]
	Qlower = [[0 for i in range(mdp.numActions)] for j in range(mdp.numStates)]
	Vupper = [Vmax for i in range(mdp.numStates)]
	Vlower = [0 for i in range(mdp.numStates)]
	best_policy = [-1 for i in range(mdp.numStates)]
	deltadeltaV = np.zeros((mdp.numActions,mdp.numActions))
	discovered_states = set([start_state])

	### Initial sampling for all state action pairs
	#### Is this needed?
	# while it < initial_iterations:
	# 	for state in range(mdp.numStates):
	# 		for act in range(mdp.numActions):
	# 			it+=1
	# 			ss, rr = mdp.simulate(state, act)
	# 			R_s_a[state][act] = (rr + R_s_a[state][act]*N_s_a[state][act])/(N_s_a[state][act]+1)
	# 			N_s_a[state][act] += 1
	# 			N_s_a_sprime[state][act][ss] += 1
	# 			# P_s_a_sprime = np.copy(N_s_a_sprime)
	# 			for s2 in range(mdp.numStates):
	# 				P_s_a_sprime[state][act][s2] = (float)(N_s_a_sprime[state][act][s2])/N_s_a[state][act]
	# samples += initial_iterations

	print P_s_a_sprime
	print "Completed initial iterations"
	
	# print Qupper, Vupper
	current_state = start_state
	### Repeat forever
	while True:

		for i in range(mdp.numStates):
			# print "For state ", i, " doing UpperP"
			for j in range(mdp.numActions):
				if(N_s_a[i][j]>0):
					P_tilda[i][j] = UpperP(i,j,delta,N_s_a_sprime[i][j],mdp.numStates,Vupper,False)

		current_state = start_state
		h=1
		# Qupper, Vupper = iteratedConvergence(Qupper,R_s_a,P_tilda,mdp.discountFactor, epsilon)
		# Qlower, Vlower = iteratedConvergence(Qlower,R_s_a,P_tilda,mdp.discountFactor, epsilon)	
		if(Vupper[start_state]-Vlower[start_state]<=epsilon):
			print Qupper[start_state],Vupper[start_state], Vlower[start_state]
			policy_lower = np.argmax(Qlower, axis=1)
			print "Iteration number ", samples
			print "Returning policy because of epsilon-convergence"
			return policy_lower

		### For all the explored or observed states, calculate del-delV 
		# explored_states = filter(lambda x: np.sum(N_s_a[x])>0, range(mdp.numStates))
		# observed_states = filter(lambda x: np.sum(N_s_a_sprime, axis=2)[x]>0, range(mdp.numStates))

		for st in list(discovered_states):
			for ac in range(mdp.numActions):
				#### Compute del del V
				deltadeltaV[st][ac] = CalculateDelDelV(st,ac,mdp,N_s_a, Qupper, start_state, P_s_a_sprime)

		#### Simulate greedily wrt deldelV
		curent_state, current_action = np.unravel_index(deltadeltaV.argmax(), deltadeltaV.shape)
		ss,rr = mdp.simulate(current_state, current_action)

		#### Add received state to the set of discovered states
		discovered_states.add(ss)
		
		### Update believed model
		R_s_a[current_state][current_action] = (rr + R_s_a[current_state][current_action]*N_s_a[current_state][current_action])/(N_s_a[current_state][current_action]+1)
		N_s_a[current_state][current_action] += 1	
		N_s_a_sprime[current_state][current_action][ss] += 1
		samples += 1
		for s2 in range(mdp.numStates):
			P_s_a_sprime[current_state][current_action][s2] = (float)(N_s_a_sprime[current_state][current_action][s2])/N_s_a[current_state][current_action]

	return best_policy


def CalculateDelDelV(state, action, mdp, N_s_a, Qupper, start_state, P):

	global Rmax, Vmax
	deldelQ = -1
	### State has not been observed
	if(N_s_a[state][action]==0):
		deldelQ = Vmax - mdp.discountFactor*Rmax/(1-mdp.discountFactor)

	else:
		#### Calculate using changed w
		pass

	policy_ouu = np.argmax(Qupper, axis=1)

	#### Get occupancy measures 
	mu_s = [0.0 for i in range(mdp.numStates)]
	prob = LpProblem("Occupancy solver",LpMinimize)
	V_s = [0 for i in range(mdp.numStates)]
	for i in range(mdp.numStates):
		mu_s[i] = LpVariable("Occupancy "+str(i))


	prob += 1, "Dummy objective function"

	for st in range(mdp.numStates):
		if(st==start_state):
			# rhs = lpSum(transitions[st][ac][sprime]*(rewards[st][ac][sprime]+gamma*V_s[sprime]) for sprime in range(numStates))
			prob += mu_s[st] == 1 + mdp.discountFactor*lpSum([mu_s[sprime]*P[sprime][policy_ouu[sprime]][st] for sprime in range(mdp.numStates)])
			# prob += V_s[st] >= sum(t*(r+2*v) for t,r,v,g in zip(transitions[st][ac],rewards[st][ac],V_s,gammaList))
		else:
			prob += mu_s[st] == mdp.discountFactor*lpSum([mu_s[sprime]*P[sprime][policy_ouu[sprime]][st] for sprime in range(mdp.numStates)])
	prob.writeLP("MDPmodel.lp")
	prob.solve()
	print "occupancy solved"

	occupancies = np.array([prob.variables()[i].varValue for i in range(mdp.numStates)])	

	return occupancies[state]*deldelQ