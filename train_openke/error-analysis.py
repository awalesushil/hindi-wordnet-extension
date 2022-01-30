import openke
from openke.config import Trainer, Tester
from openke.module.model import TransE, TransH, ComplEx, DistMult
from openke.module.loss import SoftplusLoss
from openke.module.strategy import NegativeSampling
from openke.data import TrainDataLoader, TestDataLoader


print("TransE")

# dataloader for training
train_dataloader = TrainDataLoader(
	in_path = "./benchmarks/HIWN/", 
	nbatches = 100,
	threads = 8, 
	sampling_mode = "normal", 
	bern_flag = 1, 
	filter_flag = 1, 
	neg_ent = 25,
	neg_rel = 0)

# define the model
transe = TransE(
	ent_tot = train_dataloader.get_ent_tot(),
	rel_tot = train_dataloader.get_rel_tot(),
	dim = 200, 
	p_norm = 1, 
	norm_flag = True)


# define the loss function
model = NegativeSampling(
	model = transe, 
	loss = MarginLoss(margin = 5.0),
	batch_size = train_dataloader.get_batch_size()
)

print("1-1")

# dataloader for test
test_dataloader = TestDataLoader("./benchmarks/HIWN/one_one/", "link")

# test the model
transe.load_checkpoint('./checkpoint/transe.ckpt')
tester = Tester(model = transe, data_loader = test_dataloader, use_gpu = True)
tester.run_link_prediction(type_constrain = False)

print("1-n")

# dataloader for test
test_dataloader = TestDataLoader("./benchmarks/HIWN/one_n/", "link")

# test the model
transe.load_checkpoint('./checkpoint/transe.ckpt')
tester = Tester(model = transe, data_loader = test_dataloader, use_gpu = True)
tester.run_link_prediction(type_constrain = False)

print("n-1")

# dataloader for test
test_dataloader = TestDataLoader("./benchmarks/HIWN/n_one/", "link")

# test the model
transe.load_checkpoint('./checkpoint/transe.ckpt')
tester = Tester(model = transe, data_loader = test_dataloader, use_gpu = True)
tester.run_link_prediction(type_constrain = False)

print("n-n")

# dataloader for test
test_dataloader = TestDataLoader("./benchmarks/HIWN/n_n/", "link")

# test the model
transe.load_checkpoint('./checkpoint/transe.ckpt')
tester = Tester(model = transe, data_loader = test_dataloader, use_gpu = True)
tester.run_link_prediction(type_constrain = False)


print("TransH")

# dataloader for training
train_dataloader = TrainDataLoader(
	in_path = "./benchmarks/HIWN/", 
	nbatches = 100,
	threads = 8, 
	sampling_mode = "normal", 
	bern_flag = 1, 
	filter_flag = 1, 
	neg_ent = 25,
	neg_rel = 0)

# define the model
transh = TransH(
	ent_tot = train_dataloader.get_ent_tot(),
	rel_tot = train_dataloader.get_rel_tot(),
	dim = 200, 
	p_norm = 1, 
	norm_flag = True)

# define the loss function
model = NegativeSampling(
	model = transh, 
	loss = MarginLoss(margin = 4.0),
	batch_size = train_dataloader.get_batch_size()
)

print("1-1")

# dataloader for test
test_dataloader = TestDataLoader("./benchmarks/HIWN/one_one", "link")

# test the model
transh.load_checkpoint('./checkpoint/transh.ckpt')
tester = Tester(model = transh, data_loader = test_dataloader, use_gpu = True)
tester.run_link_prediction(type_constrain = False)

print("1-n")

# dataloader for test
test_dataloader = TestDataLoader("./benchmarks/HIWN/one_n", "link")

# test the model
transh.load_checkpoint('./checkpoint/transh.ckpt')
tester = Tester(model = transh, data_loader = test_dataloader, use_gpu = True)
tester.run_link_prediction(type_constrain = False)

print("n-1")

# dataloader for test
test_dataloader = TestDataLoader("./benchmarks/HIWN/n_one", "link")

# test the model
transh.load_checkpoint('./checkpoint/transh.ckpt')
tester = Tester(model = transh, data_loader = test_dataloader, use_gpu = True)
tester.run_link_prediction(type_constrain = False)

print("n-n")

# dataloader for test
test_dataloader = TestDataLoader("./benchmarks/HIWN/n_n", "link")

# test the model
transh.load_checkpoint('./checkpoint/transh.ckpt')
tester = Tester(model = transh, data_loader = test_dataloader, use_gpu = True)
tester.run_link_prediction(type_constrain = False)

print("ComplEx")

# dataloader for training
train_dataloader = TrainDataLoader(
	in_path = "./benchmarks/HIWN/", 
	nbatches = 100,
	threads = 8, 
	sampling_mode = "normal", 
	bern_flag = 1, 
	filter_flag = 1, 
	neg_ent = 25,
	neg_rel = 0
)

# define the model
complEx = ComplEx(
	ent_tot = train_dataloader.get_ent_tot(),
	rel_tot = train_dataloader.get_rel_tot(),
	dim = 200
)

# define the loss function
model = NegativeSampling(
	model = complEx, 
	loss = SoftplusLoss(),
	batch_size = train_dataloader.get_batch_size(), 
	regul_rate = 1.0
)

print("1-1")

# dataloader for test
test_dataloader = TestDataLoader("./benchmarks/HIWN/one_one", "link")

# test the model
complEx.load_checkpoint('./checkpoint/complEx.ckpt')
tester = Tester(model = complEx, data_loader = test_dataloader, use_gpu = True)
tester.run_link_prediction(type_constrain = False)


print("1-n")

# dataloader for test
test_dataloader = TestDataLoader("./benchmarks/HIWN/one_n", "link")

# test the model
complEx.load_checkpoint('./checkpoint/complEx.ckpt')
tester = Tester(model = complEx, data_loader = test_dataloader, use_gpu = True)
tester.run_link_prediction(type_constrain = False)


print("n-1")

# dataloader for test
test_dataloader = TestDataLoader("./benchmarks/HIWN/n_one", "link")

# test the model
complEx.load_checkpoint('./checkpoint/complEx.ckpt')
tester = Tester(model = complEx, data_loader = test_dataloader, use_gpu = True)
tester.run_link_prediction(type_constrain = False)


print("n-n")

# dataloader for test
test_dataloader = TestDataLoader("./benchmarks/HIWN/n_n", "link")

# test the model
complEx.load_checkpoint('./checkpoint/complEx.ckpt')
tester = Tester(model = complEx, data_loader = test_dataloader, use_gpu = True)
tester.run_link_prediction(type_constrain = False)


print("DistMult")


# dataloader for training
train_dataloader = TrainDataLoader(
	in_path = "./benchmarks/HIWN/", 
	nbatches = 100,
	threads = 8, 
	sampling_mode = "normal", 
	bern_flag = 1, 
	filter_flag = 1, 
	neg_ent = 25,
	neg_rel = 0
)

# define the model
distmult = DistMult(
	ent_tot = train_dataloader.get_ent_tot(),
	rel_tot = train_dataloader.get_rel_tot(),
	dim = 200
)

# define the loss function
model = NegativeSampling(
	model = distmult, 
	loss = SoftplusLoss(),
	batch_size = train_dataloader.get_batch_size(), 
	regul_rate = 1.0
)

print("1-1")

# dataloader for test
test_dataloader = TestDataLoader("./benchmarks/HIWN/one_one", "link")

# test the model
distmult.load_checkpoint('./checkpoint/distmult.ckpt')
tester = Tester(model = distmult, data_loader = test_dataloader, use_gpu = True)
tester.run_link_prediction(type_constrain = False)


print("1-n")

# dataloader for test
test_dataloader = TestDataLoader("./benchmarks/HIWN/one_n", "link")

# test the model
distmult.load_checkpoint('./checkpoint/distmult.ckpt')
tester = Tester(model = distmult, data_loader = test_dataloader, use_gpu = True)
tester.run_link_prediction(type_constrain = False)


print("n-1")

# dataloader for test
test_dataloader = TestDataLoader("./benchmarks/HIWN/n_one", "link")

# test the model
distmult.load_checkpoint('./checkpoint/distmult.ckpt')
tester = Tester(model = distmult, data_loader = test_dataloader, use_gpu = True)
tester.run_link_prediction(type_constrain = False)


print("n-n")

# dataloader for test
test_dataloader = TestDataLoader("./benchmarks/HIWN/n_n", "link")

# test the model
distmult.load_checkpoint('./checkpoint/distmult.ckpt')
tester = Tester(model = distmult, data_loader = test_dataloader, use_gpu = True)
tester.run_link_prediction(type_constrain = False)