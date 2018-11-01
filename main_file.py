from dt_predicting_rain import dt
from nb_predicting_rain import nb
from hybrid_predicting_rain import hybrid

print ("NAIVE BAYES")
nb()
print "\n"
print ("DECISION TREE")
dt()
print "\n"
print ("HYBRID NAIVE BAYES AND DECISION TREE")
hybrid()
