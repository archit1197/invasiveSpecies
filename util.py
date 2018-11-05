from constants import *
import math
import numpy as np
from pulp import *
import sys

def CalculateValuePolicy(mdp, policy, H):
	total_reward = 0
	current_state = 0
	for i in range(H):
		current_action = policy[current_state]
		s, r = mdp.simulate(current_state, current_action)
		total_reward += r
		current_state = s

	return total_reward

def bestTwoActions(mdp, state, Qlower, Qupper, Qstar):
	# actionsList = []
	# actionsList.append(np.argmax(Qstar[state]))
	# a2 = np.argmax(Qupper[state])
	# if(actionsList[0]!=a2):
	# 	actionsList.append(a2)
	# else:
	# 	actionsList.append(Qupper[state].argsort()[-2:][::-1][1])
	# return actionsList

	actionsList =  Qupper[state].argsort()[-2:][::-1]
	return actionsList

def getBestPolicy(mdp, rewards, transitions):
	prob = LpProblem("MDP solver",LpMinimize)
	V_s = [0 for i in range(mdp.numStates)]
	for i in range(mdp.numStates):
		V_s[i] = LpVariable("Value for "+str(i))

	print rewards
	print transitions

	prob += lpSum(V_s[i] for i in range(mdp.numStates)), "Sum of V functions"

	for st in range(mdp.numStates):
		for ac in range(mdp.numActions):
			# rhs = lpSum(transitions[st][ac][sprime]*(rewards[st][ac][sprime]+gamma*V_s[sprime]) for sprime in range(numStates))
			prob += V_s[st] >= lpSum([transitions[st][ac][sprime]*(rewards[st][ac][sprime]+mdp.discountFactor*V_s[sprime]) for sprime in range(mdp.numStates)])
			# prob += V_s[st] >= sum(t*(r+2*v) for t,r,v,g in zip(transitions[st][ac],rewards[st][ac],V_s,gammaList))
	prob.writeLP("MDPmodel.lp")

	prob.solve()

	# print "Status:", LpStatus[prob.status]

	for v in prob.variables():
		print v.name, "=", v.varValue

	policy_final = [0 for i in range(mdp.numStates)]
	for st in range(mdp.numStates):
		maxV = -float("inf")
		maxAction = -1
		for ac in range(mdp.numActions):
			curVal = sum([transitions[st][ac][sprime]*(rewards[st][ac][sprime]+mdp.discountFactor*prob.variables()[sprime].varValue) for sprime in range(mdp.numStates)])
			if(curVal>maxV):
				maxAction = ac
				maxV = curVal

		policy_final[st] = maxAction	
	return policy_final

def hasConverged(a,b,epsilon=0.1):
	if(np.linalg.norm(np.subtract(a,b))<=epsilon):
		return True
	else:
		return False

def iteratedConvergence(Qupper, R, P, gamma, epsilon, maxIterations, eps_convergence):
	
	for i in range(maxIterations):
		# print Qupper[0]
		temp = np.copy(Qupper)
		Qmax_s = np.amax(Qupper,axis=1)
		# print "Norm ", np.linalg.norm(Qmax_s)
		Qupper = R + gamma*np.sum(P*Qmax_s, axis=2)
		Vupper = np.amax(Qupper,axis = 1)
		if(hasConverged(Qupper, temp, eps_convergence)):
			break

	return Qupper, Vupper

def wL1Confidence(N, delta, numStates):
	return math.sqrt(2*(math.log(2**numStates-2)-math.log(delta))/N)

def LowerP(state, action, delta, N_sprime, numStates, Vlower, good_turing):
	N_total = sum(N_sprime)

	P_hat_sprime = [(float)(N_sprime[i])/N_total for i in range(numStates)]
	P_tilda_sprime = [(float)(N_sprime[i])/N_total for i in range(numStates)]

	delta_w = wL1Confidence(N_total,delta, numStates)/2

	if(good_turing):
		delta_w = min(wL1Confidence(N_total,delta/2, numStates)/2,(1+math.sqrt(2))*math.sqrt(math.log(2/delta)/N_total))

	while delta_w>0:
		# print "delta_w is ", delta_w
		recipient_states = [i for i in range(numStates) if P_tilda_sprime[i]<1]
		# print "Recipient states ",recipient_states
		positive_states = [i for i in range(numStates) if P_tilda_sprime[i]>0]
		### donor state
		# print Vupper
		donor_s = max(positive_states, key=lambda x: Vlower[x])
		recipient_s = min(recipient_states, key=lambda x: Vlower[x])
		# print "Donor recipient", donor_s, recipient_s
		zeta = min(1-P_tilda_sprime[recipient_s], P_tilda_sprime[donor_s], delta_w)
		if(not P_tilda_sprime[donor_s]>sys.float_info.epsilon):
			break	
		# print "Values are : ", 1-P_tilda_sprime[recipient_s], P_tilda_sprime[donor_s], delta_w

		# print "Subtracting ", P_tilda_sprime[donor_s], zeta 
		P_tilda_sprime[donor_s] -= zeta
		P_tilda_sprime[recipient_s] += zeta

		delta_w -= zeta

	# print N_sprime, P_tilda_sprime
	return P_tilda_sprime 

def UpperP(state, action, delta, N_sprime, numStates, Vupper, good_turing):
	N_total = sum(N_sprime)

	P_hat_sprime = [(float)(N_sprime[i])/N_total for i in range(numStates)]
	P_tilda_sprime = [(float)(N_sprime[i])/N_total for i in range(numStates)]

	delta_w = wL1Confidence(N_total, delta, numStates)/2

	if(good_turing):
		delta_w = min(wL1Confidence(N_total,delta/2, numStates)/2,(1+math.sqrt(2))*math.sqrt(math.log(2/delta)/N_total))

	while delta_w>0:
		# print "delta_w is ", delta_w
		recipient_states = [i for i in range(numStates) if P_tilda_sprime[i]<1]
		# print "Recipient states ",recipient_states
		positive_states = [i for i in range(numStates) if P_tilda_sprime[i]>0]
		### donor state
		# print Vupper
		donor_s = min(positive_states, key=lambda x: Vupper[x])
		recipient_s = max(recipient_states, key=lambda x: Vupper[x])
		# print "Donor recipient", donor_s, recipient_s
		zeta = min(1-P_tilda_sprime[recipient_s], P_tilda_sprime[donor_s], delta_w)
		if(not P_tilda_sprime[donor_s]>sys.float_info.epsilon):
			break	
		# print "Values are : ", 1-P_tilda_sprime[recipient_s], P_tilda_sprime[donor_s], delta_w

		# print "Subtracting ", P_tilda_sprime[donor_s], zeta 
		P_tilda_sprime[donor_s] -= zeta
		P_tilda_sprime[recipient_s] += zeta

		delta_w -= zeta


	return P_tilda_sprime 


def CalculateDelDelV(state, action, mdp, N_s_a_sprime, Qupper, Qlower, Vupper, Vlower, start_state, P, P_tilda, P_lower_tilda, R_s_a, epsilon, delta):

	Rmax = mdp.Vmax*(1-mdp.discountFactor)
	deldelQ = -1
	### State has not been observed
	if(np.sum(N_s_a_sprime[state][action])==0):
		deldelQ = mdp.Vmax - mdp.discountFactor*Rmax/(1-mdp.discountFactor)

	else:
		#### Calculate using changed w
		# print N_s_a_sprime[state][action]
		P_tilda[state][action] = UpperP(state,action,delta,N_s_a_sprime[state][action],mdp.numStates,Vupper,False,"ddv")
		P_lower_tilda[state][action] = LowerP(state,action,delta,N_s_a_sprime[state][action],mdp.numStates,Vupper,False,"ddv")
		Quppernew, Vuppernew = iteratedConvergence(Qupper,R_s_a,P_tilda,mdp.discountFactor, epsilon, converge_iterations, epsilon_convergence)
		Qlowernew, Vlowernew = iteratedConvergence(Qlower,R_s_a,P_lower_tilda,mdp.discountFactor, epsilon, converge_iterations, epsilon_convergence)
		deldelQ = abs(Quppernew[state][action]-Qlowernew[state][action]+Qupper[state][action]-Qlower[state][action])

	policy_ouu = np.argmax(Qupper, axis=1)

	#### Get occupancy measures 
	mu_s = [0.0 for i in range(mdp.numStates)]
	prob = LpProblem("Occupancy solver",LpMinimize)
	for i in range(mdp.numStates):
		mu_s[i] = LpVariable("Occupancy "+str(i), lowBound = 0 ,upBound = mdp.discountFactor/(1-mdp.discountFactor))


	prob += 1, "Dummy objective function"

	for st in range(mdp.numStates):
		prob += mu_s[st] < mdp.discountFactor/(1-mdp.discountFactor)
		prob += mu_s[st] > 0
		if(st==start_state):
			# rhs = lpSum(transitions[st][ac][sprime]*(rewards[st][ac][sprime]+gamma*V_s[sprime]) for sprime in range(numStates))
			prob += mu_s[st] == 1 + mdp.discountFactor*lpSum([mu_s[sprime]*P[sprime][policy_ouu[sprime]][st] for sprime in range(mdp.numStates)])
			# prob += V_s[st] >= sum(t*(r+2*v) for t,r,v,g in zip(transitions[st][ac],rewards[st][ac],V_s,gammaList))
		else:
			prob += mu_s[st] == mdp.discountFactor*lpSum([mu_s[sprime]*P[sprime][policy_ouu[sprime]][st] for sprime in range(mdp.numStates)])
	prob.writeLP("MDPmodel.lp")
	prob.solve()
	# print "occupancy solved"

	occupancies = np.array([prob.variables()[i].varValue for i in range(mdp.numStates)])	
	# print P[start_state][policy_ouu[start_state]] 
	# print "occupancies", occupancies

	# print state, action, occupancies[state], deldelQ
	return occupancies[state]*deldelQ