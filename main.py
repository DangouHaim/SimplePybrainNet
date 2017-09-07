import logging
from pybrain.datasets import SupervisedDataSet
from pybrain.tools.shortcuts import buildNetwork
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.structure import TanhLayer
from pybrain.structure import SoftmaxLayer
from pybrain.tools.customxml.networkwriter import NetworkWriter
from pybrain.tools.customxml.networkreader import NetworkReader

logging.warning('start')

net = buildNetwork(2, 9, 1)
net.activate([2, 1])

ds = SupervisedDataSet(2, 1)
ds.addSample((0, 0), (0,))
ds.addSample((0, 1), (1,))
ds.addSample((1, 0), (1,))
ds.addSample((1, 1), (0,))

net = NetworkReader.readFrom('net.xml')

trainer = BackpropTrainer(net, ds)
err = 1
while err > 0.00001:
	err = trainer.train()
	logging.warning(err)

NetworkWriter.writeToFile(net, 'net.xml')

p = net.activateOnDataset(ds)

for i in p:
	logging.warning(round(i))