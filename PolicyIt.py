from constants import *
import math
import numpy as np
from util import getPolicies, UpperP, LowerP, itConvergencePolicy, getRewards, getProb

verbose = 0

def policyIt(mdp, start_state=0, epsilon=4, delta=0.1):

	policies = getPolicies(mdp.numStates, mdp.numActions)
	numPolicies = len(policies)
	counts = np.zeros((numPolicies))
	print numPolicies
	H = int((math.log(mdp.Vmax) + math.log(6.0/epsilon))/(1-mdp.discountFactor))
	
	print "Chosen value of H is : ", H

	it = 0
	samples = 0
	initial_iterations = 1*mdp.numStates*mdp.numActions
	R_s_a = np.zeros((mdp.numStates,mdp.numActions))
	N_s_a_sprime = np.zeros((mdp.numStates,mdp.numActions,mdp.numStates), dtype=np.int)
	N_s_a = np.zeros((mdp.numStates,mdp.numActions), dtype=np.int)
	P_s_a_sprime = np.zeros((mdp.numStates,mdp.numActions,mdp.numStates))
	Qupper = mdp.Vmax*np.ones((numPolicies, mdp.numStates))
	QupperMBAE = mdp.Vmax*np.ones((numPolicies, mdp.numStates))
	Qlower = np.zeros((numPolicies, mdp.numStates))
	Qstar = (mdp.Vmax/2)*np.ones((numPolicies, mdp.numStates))
	QstarMBAE = (mdp.Vmax/2)*np.ones((numPolicies, mdp.numStates))
	QlowerMBAE = np.zeros((numPolicies, mdp.numStates))
	P_tilda = np.zeros((numPolicies, mdp.numStates,mdp.numStates))
	P_lower_tilda = np.zeros((numPolicies, mdp.numStates,mdp.numStates))
	VlowerMBAE = np.zeros((numPolicies, mdp.numStates))
	VupperMBAE = mdp.Vmax*np.ones((numPolicies, mdp.numStates))
	Vstar = (mdp.Vmax/2)*np.ones((numPolicies, mdp.numStates))

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

	ff = open(mdp.filename+'-policy.txt', 'w+')
	while True:
		
		# print counts
		for p in range(numPolicies):
			# print "Policy Number : ", p
			current_policy = policies[p]

			for i in range(mdp.numStates):
				# print "For state ", i, " doing UpperP"
				if(N_s_a[i][current_policy[i]]>0):
					P_tilda[p][i] = UpperP(i,current_policy[i],delta,N_s_a_sprime[i][current_policy[i]],mdp.numStates,Qupper[p],False)
					P_lower_tilda[p][i] = LowerP(i,current_policy[i],delta,N_s_a_sprime[i][current_policy[i]],mdp.numStates,Qlower[p],False)
				# import pdb; pdb.set_trace()

			Qupper[p] = itConvergencePolicy(Qupper[p],getRewards(R_s_a, current_policy),P_tilda[p],mdp.discountFactor, epsilon, converge_iterations, epsilon_convergence)
			Qlower[p] = itConvergencePolicy(Qlower[p],getRewards(R_s_a, current_policy),P_lower_tilda[p],mdp.discountFactor, epsilon, converge_iterations, epsilon_convergence)	
			Qstar[p] = itConvergencePolicy(Qstar[p],getRewards(R_s_a, current_policy),getProb(P_s_a_sprime, current_policy),mdp.discountFactor, epsilon, converge_iterations, epsilon_convergence)	

			# print "mbie bounds calculated!"
			for internal in range(converge_iterations):
				
				oldQlowerMBAE = np.copy(QlowerMBAE[p][start_state])
				for state in range(mdp.numStates):
					# for act in range(mdp.numActions):
					act = current_policy[state]
						# Calculations for QupperMBAE and QlowerMBAE
					firstterm = R_s_a[state][act]
					secondterm = mdp.discountFactor*np.sum(VupperMBAE[p]*(P_s_a_sprime[state][act]))
					lower_secondterm = mdp.discountFactor*np.sum(VlowerMBAE[p]*(P_s_a_sprime[state][act]))
					star_secondterm = mdp.discountFactor*np.sum(Vstar[p]*(P_s_a_sprime[state][act]))
					thirdterm = mdp.Vmax*math.sqrt((math.log(c*(samples**2)*mdp.numStates*1)-math.log(delta))/N_s_a[state][act])
					QupperMBAE[p][state] = firstterm + secondterm + thirdterm
					QlowerMBAE[p][state] = firstterm + lower_secondterm - thirdterm
					QstarMBAE[p][state] = firstterm + star_secondterm
					VupperMBAE[p][state] = QupperMBAE[p][state]
					VlowerMBAE[p][state] = QlowerMBAE[p][state]
					Vstar[p][state] = QstarMBAE[p][state]
				if(np.linalg.norm(oldQlowerMBAE-QlowerMBAE[p][start_state])<=epsilon_convergence):
					break

		
		# import pdb; pdb.set_trace()
		policy1Index = np.argmax(QstarMBAE[:,start_state])
		policy2choices = QupperMBAE[:,start_state].argsort()[::-1]
		if(policy2choices[0]==policy1Index):
			policy2Index = policy2choices[1]
		else:
			policy2Index = policy2choices[0]

		h=0
		policy1 = policies[policy1Index]
		policy2 = policies[policy2Index]
		state = start_state
		if (samples%10000)<300:
			if(verbose==0):
				# ff.write('Showing Qupper, Qstar \n')
				# np.savetxt(ff, (QupperMBAE[:,start_state],Qstar[:,start_state]), fmt="%f")
				# np.savetxt(ff, QupperMBAE[:,start_state], delimiter=',')
				# np.savetxt(ff, Qstar[:,start_state], delimiter=',')
				# print QupperMBAE[:,start_state]
				# print Qstar[:,start_state]
				ff.write(str(samples))
				ff.write('\t')
				ff.write(str(QupperMBAE[policy2Index][start_state]-QlowerMBAE[policy1Index][start_state]))#-epsilon*(1-mdp.discountFactor)/2 
				print QupperMBAE[0][0]
				ff.write('\n')
			else:
				print samples
				print QupperMBAE[:,start_state], QlowerMBAE[:,start_state]
			# np.savetxt(ff, (policies[policy1Index]), fmt="%d")
		counts[policy1Index] += 1
		counts[policy2Index] += 1
		while h<H:
			act = policy1[state]
			# print "------>",current_state, current_action
			ss, rr = mdp.simulate(state, act)
			samples+=1
			R_s_a[state][act] = (rr + R_s_a[state][act]*N_s_a[state][act])/(N_s_a[state][act]+1)
			N_s_a[state][act] += 1
			N_s_a_sprime[state][act][ss] += 1
			# P_s_a_sprime = np.copy(N_s_a_sprime)
			for s2 in range(mdp.numStates):
				P_s_a_sprime[state][act][s2] = (float)(N_s_a_sprime[state][act][s2])/N_s_a[state][act]
			state = ss
			h+=1

		h=0

		state = start_state
		# print "episode : "
		while h<H:
			# print state, 
			act = policy2[state]
			ss, rr = mdp.simulate(state, act)
			samples+=1
			R_s_a[state][act] = (rr + R_s_a[state][act]*N_s_a[state][act])/(N_s_a[state][act]+1)
			N_s_a[state][act] += 1
			N_s_a_sprime[state][act][ss] += 1
			# P_s_a_sprime = np.copy(N_s_a_sprime)
			for s2 in range(mdp.numStates):
				P_s_a_sprime[state][act][s2] = (float)(N_s_a_sprime[state][act][s2])/N_s_a[state][act]
			state = ss
			h+=1

		if (samples%1000)<100:
			if(QupperMBAE[policy2Index][start_state]-QlowerMBAE[policy1Index][start_state]-epsilon*(1-mdp.discountFactor)/2<0):
				print Qupper[policy2Index][start_state],Qstar[policy1Index][start_state],epsilon*(1-mdp.discountFactor)/2
				print "Epsilon condition reached at ",samples, " samples"
				# return policy1
			else:
				# print QupperMBAE[policy2Index][start_state],QstarMBAE[policy1Index][start_state],epsilon*(1-mdp.discountFactor)/2
				pass
			# print "ends here"

	ff.close()

