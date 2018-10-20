from constants import *
import sys
from MDPclass import myMDP
from FIECHTER import FeichterPolicy
from RR import RoundRobin
from LUCB import LUCBStopping
from MBIE import mbie
from DDVUpper import ddvupper

def main(argv):
	# print "Executing MDP"
	print argv[1][argv[1].find('/')+1:]
	mdpname = argv[1][argv[1].find('/')+1:]
	lines = [line.rstrip('\n') for line in open(argv[1])]
	print argv[2]

	global Rmax
	global Vmax
	numStates = int(lines[0])
	numActions = int(lines[1])
	rewards = lines[2].split()
	transitionProbabilities = lines[3].split()
	print lines[4]
	discountFactor = float(lines[4])
	filename = mdpname[mdpname.find('-')+1:mdpname.find('.')]
	theMDP = myMDP(numStates, numActions, rewards, transitionProbabilities, discountFactor, filename)
	theMDP.printMDP()
	eps = eps_values[mdpname]
	if(argv[2]=="uniform"):
		print "Doing naive uniform sampling"
		print UniformSampling(5, theMDP)
	elif(argv[2]=="fiechter"):
		print "Doing Fiechter algorithm"
		print FeichterPolicy(theMDP, 0, eps)
	elif(argv[2]=="rr"):
		print "Doing Round robin"
		print RoundRobin(theMDP, 0, eps)
	elif(argv[2]=="lucb"):
		print "Doing LUCB type algorithm"
		print LUCBStopping(theMDP, 0, eps)
	elif(argv[2]=="mbie"):
		print "Doing MBIE-reset algorithm"
		print mbie(theMDP, 0, eps)
	elif(argv[2]=="ddv-upper"):
		print "Doing DDV Upper algorithm"
		print ddvupper(theMDP, 0, eps)


def UniformSampling(times, mdp):
	"""Adopt a uniform sampling strategy and return the policy"""
	policy_answer = []
	for s in range(mdp.numStates):
		avg_rewards = [0 for i in range(mdp.numActions)]
		for a in range(mdp.numActions):
			for i in range(times):
				newstate, reward = mdp.simulate(s, a)
				avg_rewards[a] += int(reward)
			avg_rewards[a] /= times
		policy_answer.append(avg_rewards.index(max(avg_rewards)))
	return policy_answer

if __name__ == '__main__':
	if(len(sys.argv)<3):
		print "Usage : python MDPclass.py <mdpfile> <algorithm> [uniform, fiechter, lucb] <epsilon>"
	else:
		main(sys.argv)