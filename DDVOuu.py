from constants import *
import math
import numpy as np
from pulp import *
from util import LowerP, UpperP, CalculateDelDelV, iteratedConvergence, bestTwoActions

def ddvouu(mdp, start_state=0, epsilon=4, delta=0.1):

	initial_iterations = 1000*mdp.numStates*mdp.numActions
	### Estimate the horizon based on Fiechter
	
	c = 1
	it=0
	samples = 0

	### Calculating m based on the parameters

	first_term = mdp.numStates/(epsilon**2*(1-mdp.discountFactor)**4)
	second_term = math.log(mdp.numStates*mdp.numActions/(epsilon*(1-mdp.discountFactor)*delta))/(epsilon**2*(1-mdp.discountFactor)**4)
	m = c*(first_term+second_term)
	delta = delta/(mdp.numStates*mdp.numActions*m)
	print "Chosen value of m is :", m
	N_s_a = np.zeros((mdp.numStates,mdp.numActions), dtype=np.int)
	N_s_a_sprime = np.zeros((mdp.numStates,mdp.numActions,mdp.numStates), dtype=np.int)
	P_s_a_sprime = np.zeros((mdp.numStates,mdp.numActions,mdp.numStates))
	P_tilda = np.zeros((mdp.numStates,mdp.numActions,mdp.numStates))
	P_lower_tilda = np.zeros((mdp.numStates,mdp.numActions,mdp.numStates))
	R_s_a = np.zeros((mdp.numStates,mdp.numActions))
	Qupper = mdp.Vmax*np.ones((mdp.numStates,mdp.numActions))
	QupperMBAE = mdp.Vmax*np.ones((mdp.numStates,mdp.numActions))
	Qlower = np.zeros((mdp.numStates,mdp.numActions))
	QlowerMBAE = np.zeros((mdp.numStates,mdp.numActions))
	Vupper = mdp.Vmax*np.ones((mdp.numStates))
	VupperMBAE = mdp.Vmax*np.ones((mdp.numStates))
	Vlower = np.zeros((mdp.numStates))
	VlowerMBAE = np.zeros((mdp.numStates))
	best_policy = (-1)*np.ones((mdp.numStates), dtype=np.int)
	deltadeltaV = np.zeros((mdp.numStates,mdp.numActions))
	discovered_states = set([start_state])

	## Initial sampling for all state action pairs
	### Is this needed?
	while it < initial_iterations:
		for state in range(mdp.numStates):
			for act in range(mdp.numActions):
				it+=1
				ss, rr = mdp.simulate(state, act)
				R_s_a[state][act] = (rr + R_s_a[state][act]*N_s_a[state][act])/(N_s_a[state][act]+1)
				N_s_a[state][act] += 1
				N_s_a_sprime[state][act][ss] += 1
				# P_s_a_sprime = np.copy(N_s_a_sprime)
				for s2 in range(mdp.numStates):
					P_s_a_sprime[state][act][s2] = (float)(N_s_a_sprime[state][act][s2])/N_s_a[state][act]
	samples += initial_iterations

	print P_s_a_sprime
	print "Completed initial iterations"

	sys.stdout = open(mdp.filename+'-ddv.txt', 'w+')
	ff = open(mdp.filename+'-ddv-samples.txt', 'w+')
	
	# print Qupper, Vupper
	current_state = start_state
	### Repeat forever
	while True:

		for i in range(mdp.numStates):
			# print "For state ", i, " doing UpperP"
			for j in range(mdp.numActions):
				if(N_s_a[i][j]>0):
					P_tilda[i][j] = UpperP(i,j,delta,N_s_a_sprime[i][j],mdp.numStates,Vupper,False)
					P_lower_tilda[i][j] = LowerP(i,j,delta,N_s_a_sprime[i][j],mdp.numStates,Vupper,False)

		Qupper, Vupper = iteratedConvergence(Qupper,R_s_a,P_tilda,mdp.discountFactor, epsilon, converge_iterations, epsilon_convergence)
		Qlower, Vlower = iteratedConvergence(Qlower,R_s_a,P_lower_tilda,mdp.discountFactor, epsilon, converge_iterations, epsilon_convergence)	

		current_state = start_state

		# print Qupper
		if(Vupper[start_state]-Vlower[start_state]<=0):
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
				deltadeltaV[st][ac] = CalculateDelDelV(st,ac,mdp,N_s_a_sprime, Qupper, Qlower, Vupper, Vlower, start_state, P_s_a_sprime, P_tilda, P_lower_tilda, R_s_a, epsilon, delta)

		# print deltadeltaV
		#### Simulate greedily wrt deldelV
		curent_state, current_action = np.unravel_index(deltadeltaV.argmax(), deltadeltaV.shape)
		ss,rr = mdp.simulate(current_state, current_action)
		# print "Choosing ", current_state, current_action
		# print P_s_a_sprime
		#### Add received state to the set of discovered states
		discovered_states.add(ss)
		
		### Update believed model
		R_s_a[current_state][current_action] = (rr + R_s_a[current_state][current_action]*N_s_a[current_state][current_action])/(N_s_a[current_state][current_action]+1)
		N_s_a[current_state][current_action] += 1	
		N_s_a_sprime[current_state][current_action][ss] += 1
		samples += 1
		for s2 in range(mdp.numStates):
			# print current_state, current_action, s2, N_s_a_sprime[current_state][current_action][s2], N_s_a[current_state][current_action]
			P_s_a_sprime[current_state][current_action][s2] = (float)(N_s_a_sprime[current_state][current_action][s2])/N_s_a[current_state][current_action]
		if(samples%10000==0):
				acList = bestTwoActions(mdp, start_state, Qlower, Qupper)
				print samples, (QupperMBAE[start_state][acList[1]]-QlowerMBAE[start_state][acList[0]])/epsilon 
				np.savetxt(ff, N_s_a, delimiter=',')
				ff.write('\n')

		### Calculating MBAE bounds
		for internal in range(converge_iterations):
			oldQlower = np.copy(QlowerMBAE[start_state])
			for state in range(mdp.numStates):
				for act in range(mdp.numActions):
					# Calculations for Qupper and Qlower
					firstterm = R_s_a[state][act]
					secondterm = mdp.discountFactor*np.sum(VupperMBAE*(N_s_a_sprime[state][act]/N_s_a[state][act]))
					#secondterm = mdp.discountFactor*sum(Vupper[ss]*N_s_a_sprime[state][act][ss]/N_s_a[state][act] for ss in range(mdp.numStates))  
					lower_secondterm = mdp.discountFactor*np.sum(VlowerMBAE*(N_s_a_sprime[state][act]/N_s_a[state][act]))
					#lower_secondterm = mdp.discountFactor*sum(Vlower[ss]*N_s_a_sprime[state][act][ss]/N_s_a[state][act] for ss in range(mdp.numStates))  
					thirdterm = mdp.Vmax*math.sqrt((math.log(c*(samples**2)*mdp.numStates*mdp.numActions)-math.log(delta))/N_s_a[state][act])
					#Qupper[state][act] = (float)(sum(rewards_s_a_sprime[state][act][ss] for ss in range(mdp.numStates))/N_s_a[state][act]) + secondterm + thirdterm
					QupperMBAE[state][act] = firstterm + secondterm + thirdterm
					Qlower[state][act] = firstterm + lower_secondterm - thirdterm

					# Calculation for Vstar
					# t = (float)N_s_a_sprime[state][act][stateprime]/N_s_a[state][act]
					# val = t*(rewards_s_a[state][act][stateprime]+mdp.discountFactor*Vstar[stateprime])
				VupperMBAE[state] = np.amax(QupperMBAE[state])
				VlowerMBAE[state] = np.amax(QlowerMBAE[state])
			if(np.linalg.norm(oldQlower-QlowerMBAE[start_state])<=epsilon_convergence):
				# print "Stopping with ", internal, "iterations"
				break

		

	return best_policy
