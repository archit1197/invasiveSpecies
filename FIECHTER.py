from constants import *
import math
import numpy as np
import sys
from util import getBestPolicy, bestTwoActions

def FeichterPolicy(mdp, start_state=0, epsilon=1, delta=0.1):
	global c
	# orig_stdout = sys.stdout
	# f = open('Fiechter-m01.txt', 'w')
	# sys.stdout = f
	
	##### Initialisation
	print mdp.Vmax, 6/epsilon, mdp.discountFactor
	H = int((math.log(mdp.Vmax) + math.log(6.0/epsilon))/(1-mdp.discountFactor))
	
	print "Chosen value of H is : ", H
	N_h_s_a	= np.zeros((H,mdp.numStates,mdp.numActions))
	N_h_s_a_s_prime	= np.zeros((H,mdp.numStates,mdp.numActions,mdp.numStates), dtype=np.int)
	rewards_s_a_sprime	= np.zeros((mdp.numStates,mdp.numActions,mdp.numStates))
	P_h_s_a_s_prime = np.zeros((H,mdp.numStates,mdp.numActions,mdp.numStates))
	policy_h_s = np.zeros((H,mdp.numStates), dtype=np.int)
	d_h_policy_s = np.zeros((H+1,mdp.numStates))
	dmax = 12*mdp.Vmax/(epsilon*(1-mdp.discountFactor))
	converge_iterations = 10000
	epsilon_convergence = 1e-4

	Qlower = np.zeros((mdp.numStates,mdp.numActions))
	Qupper = mdp.Vmax*np.ones((mdp.numStates,mdp.numActions))
	Vupper = mdp.Vmax*np.ones((mdp.numStates))
	Vlower = np.zeros((mdp.numStates))
	sampled_frequency_s_a = np.zeros((mdp.numStates,mdp.numActions))
	N_s_a_sprime = np.zeros((mdp.numStates,mdp.numActions,mdp.numStates))
	it=0
	samples=0
	initial_iterations = 1*mdp.numStates*mdp.numActions

	### Initial sampling for all state action pairs
	while it < initial_iterations:
		for state in range(mdp.numStates):
			for act in range(mdp.numActions):
				it+=1
				s_prime, r = mdp.simulate(state, act)
				rewards_s_a_sprime[state][act][s_prime] += r
				sampled_frequency_s_a[state][act] += 1
				N_s_a_sprime[state][act][s_prime] += 1
	
	#### For starting the while loop below
	iteration = 1

	sys.stdout = open(mdp.filename+'-fiechter.txt', 'w+')
	ff = open(mdp.filename+'-fiechter-samples.txt', 'w+')

	#### Exploration
	while d_h_policy_s[0][start_state]>2/(1-mdp.discountFactor) or iteration==1:
		# print d_h_policy_s[0][start_state], " > ", 2/(1-mdp.discountFactor)
		# print policy_h_s[0]
		h=0
		current_state = start_state
		while h<H:
			current_action = policy_h_s[h][current_state]
			# print "------>",current_state, current_action
			s_prime, r = mdp.simulate(current_state, current_action)
			N_h_s_a[h][current_state][current_action] += 1
			rewards_s_a_sprime[current_state][current_action][s_prime] += r
			N_h_s_a_s_prime[h][current_state][current_action][s_prime] += 1
			N_s_a_sprime[current_state][current_action][s_prime] += 1
			sampled_frequency_s_a[current_state][current_action] += 1
			for s2 in range(mdp.numStates):
				P_h_s_a_s_prime[h][current_state][current_action][s2] = N_h_s_a_s_prime[h][current_state][current_action][s2]/N_h_s_a[h][current_state][current_action]
			h += 1
			current_state = s_prime
			samples += 1
			if(samples%10000==0):
				acList = bestTwoActions(mdp, start_state, Qlower, Qupper)
				print samples, (Qupper[start_state][acList[1]]-Qlower[start_state][acList[0]])/epsilon 
				np.savetxt(ff, sampled_frequency_s_a, delimiter=',')
				ff.write('\n')
				# print samples, d_h_policy_s[0][start_state]-2/(1-mdp.discountFactor) 

		# Compute new policy dynamic program
		e_s_a = np.zeros((mdp.numStates, mdp.numActions))
		for h in range(H-1, -1, -1):
			for state in range(mdp.numStates):
				current_max = -float("inf")
				argmax_action = -1
				for act in range(mdp.numActions):
					if(N_h_s_a[h][state][act]==0):
						e_s_a[state][act] = dmax
					else:
						sqterm = (2*math.log(4*H*mdp.numStates*mdp.numActions)-2*math.log(delta)) / N_h_s_a[h][state][act]
						summation = np.sum((N_h_s_a_s_prime[h][state][act]/N_h_s_a[h][state][act])*d_h_policy_s[h+1])
						secondterm = mdp.discountFactor*summation
						e_s_a[state][act] = min(dmax, 6*mdp.Vmax*(math.sqrt(sqterm))/(epsilon*(1-delta)) + secondterm)

				policy_h_s[h][state] = np.argmax(e_s_a[state])
				d_h_policy_s[h][state] = np.amax(e_s_a[state])


		# Compute MBAE Qupper and Qlower bounds
		for internal in range(converge_iterations):
			oldQlower = np.copy(Qlower[start_state])
			for state in range(mdp.numStates):
				for act in range(mdp.numActions):
				# Calculations for Qupper and Qlower
					# Calculations for Qupper and Qlower
					firstterm = np.sum(rewards_s_a_sprime[state][act])/sampled_frequency_s_a[state][act]
					secondterm = mdp.discountFactor*np.sum(Vupper*(N_s_a_sprime[state][act]/sampled_frequency_s_a[state][act]))
					#secondterm = mdp.discountFactor*sum(Vupper[ss]*N_s_a_sprime[state][act][ss]/sampled_frequency_s_a[state][act] for ss in range(mdp.numStates))  
					lower_secondterm = mdp.discountFactor*np.sum(Vlower*(N_s_a_sprime[state][act]/sampled_frequency_s_a[state][act]))
					#lower_secondterm = mdp.discountFactor*sum(Vlower[ss]*N_s_a_sprime[state][act][ss]/sampled_frequency_s_a[state][act] for ss in range(mdp.numStates))  
					thirdterm = mdp.Vmax*math.sqrt((math.log(c*(samples**2)*mdp.numStates*mdp.numActions)-math.log(delta))/sampled_frequency_s_a[state][act])
					#Qupper[state][act] = (float)(sum(rewards_s_a_sprime[state][act][ss] for ss in range(mdp.numStates))/sampled_frequency_s_a[state][act]) + secondterm + thirdterm
					Qupper[state][act] = firstterm + secondterm + thirdterm
					Qlower[state][act] = firstterm + lower_secondterm - thirdterm

					# Calculation for Vstar
					# t = (float)N_s_a_sprime[state][act][stateprime]/sampled_frequency_s_a[state][act]
					# val = t*(rewards_s_a[state][act][stateprime]+mdp.discountFactor*Vstar[stateprime])
				Vupper[state] = np.amax(Qupper[state])
				Vlower[state] = np.amax(Qlower[state])
			if(np.linalg.norm(oldQlower-Qlower[start_state])<=epsilon_convergence):
				# print "Stopping with ", internal, "iterations"
				break
			
		iteration += 1
		# print d_h_policy_s
	# sys.stdout = orig_stdout
	# f.close()
	print iteration
	return getBestPolicy(mdp,rewards_s_a_sprime,P_h_s_a_s_prime[0])
	# return policy_h_s[0]