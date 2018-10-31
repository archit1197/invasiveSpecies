from constants import *
import math
import numpy as np
import sys
from util import iteratedConvergence, wL1Confidence, UpperP, bestTwoActions

def mbie(mdp, start_state=0, epsilon=4, delta=0.1):

	global c
	initial_iterations = 1*mdp.numStates*mdp.numActions
	### Estimate the horizon based on Fiechter
	H = int((math.log(mdp.Vmax) + math.log(6.0/epsilon))/(1-mdp.discountFactor))
	it=0
	samples = 0

	### Calculating m based on the parameters

	first_term = mdp.numStates/(epsilon**2*(1-mdp.discountFactor)**4)
	second_term = math.log(mdp.numStates*mdp.numActions/(epsilon*(1-mdp.discountFactor)*delta))/(epsilon**2*(1-mdp.discountFactor)**4)
	m = c*(first_term+second_term)
	print "Chosen value of m is :", m
	N_s_a = np.zeros((mdp.numStates,mdp.numActions))
	N_s_a_sprime = np.zeros((mdp.numStates,mdp.numActions,mdp.numStates))
	P_s_a_sprime = np.zeros((mdp.numStates,mdp.numActions,mdp.numStates))
	P_tilda = np.zeros((mdp.numStates,mdp.numActions,mdp.numStates))
	R_s_a = np.zeros((mdp.numStates,mdp.numActions))
	Qupper = mdp.Vmax*np.ones((mdp.numStates,mdp.numActions))
	QupperMBAE = mdp.Vmax*np.ones((mdp.numStates,mdp.numActions))
	Qlower = np.zeros((mdp.numStates,mdp.numActions))
	Qstar = (mdp.Vmax/2)*np.ones((mdp.numStates,mdp.numActions))
	Vupper = mdp.Vmax*np.ones((mdp.numStates))
	VupperMBAE = mdp.Vmax*np.ones((mdp.numStates))
	Vlower = np.zeros((mdp.numStates))
	Vstar = (mdp.Vmax/2)*np.ones((mdp.numStates))
	best_policy = (-1)*np.ones((mdp.numStates), dtype=np.int)
	converge_iterations = 10000
	epsilon_convergence = 1e-4

	### Initial sampling for all state action pairs
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
	Qupper, Vupper = iteratedConvergence(Qupper,R_s_a,P_s_a_sprime,mdp.discountFactor, epsilon, converge_iterations, epsilon_convergence)
	print Qupper, "Qupper"
	# print Qupper, Vupper
	current_state = start_state
	### Repeat forever

	sys.stdout = open(mdp.filename+'-mbie.txt', 'w+')
	ff = open(mdp.filename+'-mbie-samples.txt', 'w+')

	while samples<MAX_ITERATION_LIMIT:

		current_state = start_state
		h=1
		while h<=H:
			if(samples%10000==0):
				acList = bestTwoActions(mdp, start_state, Qlower, Qupper)
				print samples, (QupperMBAE[start_state][acList[1]]-Qlower[start_state][acList[0]])/epsilon 
				np.savetxt(ff, N_s_a, delimiter=',')
				ff.write('\n')
			for i in range(mdp.numStates):
				# print "For state ", i, " doing UpperP"
				for j in range(mdp.numActions):
					P_tilda[i][j] = UpperP(i,j,delta,N_s_a_sprime[i][j],mdp.numStates,Vupper,False)
			# print "Starting iterating"
			# print Qupper
			# return 2
			Qupper, Vupper = iteratedConvergence(Qupper,R_s_a,P_tilda,mdp.discountFactor,epsilon, converge_iterations, epsilon_convergence)
			current_action = np.argmax(Qupper[current_state])
			best_policy[current_state] = current_action
			if(N_s_a[current_state][current_action]<m):
				for t in range(1):
					ss,rr = mdp.simulate(current_state, current_action)
					R_s_a[current_state][current_action] = (rr + R_s_a[current_state][current_action]*N_s_a[current_state][current_action])/(N_s_a[current_state][current_action]+1)
					N_s_a[current_state][current_action] += 1
					N_s_a_sprime[current_state][current_action][ss] += 1
					samples += 1
				for s2 in range(mdp.numStates):
					# print current_state, current_action, s2, N_s_a_sprime[current_state][current_action][s2], N_s_a[current_state][current_action]
					P_s_a_sprime[state][act][s2] = (float)(N_s_a_sprime[state][act][s2])/N_s_a[state][act]
				current_state = ss

			else:
				# print P_s_a_sprime[current_state][current_action]
				# print np.sum(P_s_a_sprime[current_state][current_action])
				# print N_s_a[current_state][current_action]
				current_state = np.random.choice(np.arange(mdp.numStates), p=P_s_a_sprime[current_state][current_action]/np.sum(P_s_a_sprime[current_state][current_action]))

		
			h += 1
		
		# Compute MBAE Qupper and Qlower bounds
		for internal in range(converge_iterations):
			oldQlower = np.copy(Qlower[start_state])
			for state in range(mdp.numStates):
				for act in range(mdp.numActions):
					# Calculations for Qupper and Qlower
					firstterm = R_s_a[state][act]
					secondterm = mdp.discountFactor*np.sum(VupperMBAE*(N_s_a_sprime[state][act]/N_s_a[state][act]))
					#secondterm = mdp.discountFactor*sum(Vupper[ss]*N_s_a_sprime[state][act][ss]/N_s_a[state][act] for ss in range(mdp.numStates))  
					lower_secondterm = mdp.discountFactor*np.sum(Vlower*(N_s_a_sprime[state][act]/N_s_a[state][act]))
					star_secondterm = mdp.discountFactor*np.sum(Vstar*(N_s_a_sprime[state][act]/N_s_a[state][act]))
					#lower_secondterm = mdp.discountFactor*sum(Vlower[ss]*N_s_a_sprime[state][act][ss]/N_s_a[state][act] for ss in range(mdp.numStates))  
					thirdterm = mdp.Vmax*math.sqrt((math.log(c*(samples**2)*mdp.numStates*mdp.numActions)-math.log(delta))/N_s_a[state][act])
					#Qupper[state][act] = (float)(sum(rewards_s_a_sprime[state][act][ss] for ss in range(mdp.numStates))/N_s_a[state][act]) + secondterm + thirdterm
					QupperMBAE[state][act] = firstterm + secondterm + thirdterm
					Qlower[state][act] = firstterm + lower_secondterm - thirdterm
					Qstar[state][act] = firstterm + star_secondterm
					# Calculation for Vstar
					# t = (float)N_s_a_sprime[state][act][stateprime]/N_s_a[state][act]
					# val = t*(rewards_s_a[state][act][stateprime]+mdp.discountFactor*Vstar[stateprime])
				VupperMBAE[state] = np.amax(QupperMBAE[state])
				Vlower[state] = np.amax(Qlower[state])
				Vstar[state] = np.amax(Qstar[state])
			if(np.linalg.norm(oldQlower-Qlower[start_state])<=epsilon_convergence):
				# print "Stopping with ", internal, "iterations"
				break		

		# print np.linalg.norm(QupperMBAE)


	return best_policy
