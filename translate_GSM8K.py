import json
import datasets
import yaml

with open("config.yml", 'r') as file:
    config = yaml.safe_load(file)

dataset = getattr(datasets, "GSM8K")(config)

translated_gsm8k_prompt = """问：小树林里有15棵树。树林工人今天将在树林里种树。他们完成后，将有21棵树。林场工人今天种了多少棵树？
答：原来有15棵树。然后又种了一些之后，就有21棵了。所以肯定有21-5=6。答案是6。

问：如果停车场里有3辆汽车，又有2辆汽车到达，那么停车场里有多少辆汽车？
答：原来有3辆汽车。又有2辆汽车到达。3 + 2 = 5. 答案是5。

问：利亚有32块巧克力，她姐姐有42块。如果她们吃了35块，她们总共还剩下多少块？
答：最初，利亚有32块巧克力。她的姐姐有42块。所以他们总共有32+42=74块。吃了35块后，他们有74-35=39。答案是39。

问：杰森有20根棒棒糖。他给了Denny一些棒棒糖。现在杰森有12根棒棒糖。杰森给了丹尼多少个棒棒糖？
答：杰森开始有20根棒棒糖。然后他给了丹尼一些后有12个。所以他给了Denny 20 - 12 = 8，答案是8。

问：肖恩有五个玩具。圣诞节时，他从爸爸妈妈那里各得到了两个玩具。他现在有多少个玩具？
答：Shawn开始有5个玩具。如果他从爸爸妈妈那里各得到两个玩具，那么就多了4个玩具。5 + 4 = 9. 答案是9。

问：机房里有9台电脑。从星期一到星期四，每天再安装五台电脑。现在机房里有多少台电脑？
答：原来有9台电脑。在4天里，每天增加5台电脑。所以增加了5+4=20台电脑。9+20等于29。答案是29。

问：迈克尔有58个高尔夫球。在星期二，他丢了23个高尔夫球。星期三，他又丢了两个。周三结束时，他有多少个高尔夫球？
答：迈克尔一开始有58个高尔夫球。周二失去23个后，他有58-23=35。再输掉2个后，他有35-2=33个高尔夫球。答案是33。

问：奥利维亚有23美元，她买了五个百吉饼，每个3美元。她还剩下多少钱？
答：奥利维亚有23美元。5个百吉饼，每个3美元，将是5 x 3 = 15美元。所以她还剩下23-15美元。23-15是8，答案是8。"""

prompt = translated_gsm8k_prompt.split("\n\n")
with open('few-shot-exemplars/few-shot-cot-exemplars-cn.json', 'w', encoding='utf-8') as f:
    json.dump({"GSM8K": prompt}, f, ensure_ascii=False, indent=4)


original_gsm8k_train = "\n\n".join(dataset.questions[:20])
print(original_gsm8k_train)

translated_gsm8k_train = """珍妮特的鸭子每天产16个蛋。她每天早上吃三个作为早餐，每天用四个为她的朋友烤松饼。她每天在农贸市场出售剩余的鸭蛋，每个新鲜鸭蛋2美元。她每天在农贸市场能赚多少美元？

一件长袍需要2根蓝色纤维和一半的白色纤维。 它总共需要多少根纤维？

乔希决定尝试翻转房子。 他以80,000美元买了一栋房子，然后投入50,000美元进行维修。 这使房子的价值增加了150%。 他赚了多少钱？

詹姆斯决定每周跑3次短跑。 他每次冲刺都跑60米。 他一周总共跑了多少米？

每天，温迪给她的每只鸡喂三杯混合鸡饲料，其中含有种子、黄粉虫和蔬菜，以帮助它们保持健康。 她分三餐给鸡喂食。早上，她给她的鸡群喂15杯饲料。 下午，她又给鸡群喂了25杯饲料。 如果文迪的鸡群规模为20只，那么她在一天的最后一餐中需要给鸡群多少杯饲料？

Kylar去商店为他的新公寓买眼镜。一个杯子的价格是5美元，但每第二个杯子的价格只有60%。Kylar想买16个杯子。他需要支付多少钱？

图卢兹的羊是查尔斯顿的两倍。查尔斯顿的羊数是西雅图的4倍。如果西雅图有20只羊，图卢兹、查尔斯顿和西雅图一共有多少只羊？

Carla正在下载一个200GB的文件。通常情况下，她可以每分钟下载2GB，但在下载过程中的40%，Windows强制重启以安装更新，这需要20分钟。然后卡拉不得不从头开始重新启动下载。下载文件需要多少负荷？

约翰以60英里/小时的速度开了3个小时，然后掉头就走，因为他意识到他把非常重要的东西忘在了家里。 他试图在4个小时内回家，但前2个小时是在停滞不前的交通中度过的。 接下来的半小时，他以每小时30英里的速度行驶，然后以每小时80英里的速度行驶了4小时的剩余时间。 这4个小时结束后，他离家有多远？

伊丽莎每周工作的前40个小时，每小时的工资是10美元。她还可以得到1.2倍于正常时薪的加班费。如果Eliza本周工作了45小时，她本周的收入是多少？

一个新的程序在第一个月有60次下载。第二个月的下载量是第一个月的三倍，但在第三个月又减少了30%。在这三个月中，该程序总共有多少次下载？

Toula去面包店买了各种类型的糕点。她买了3打甜甜圈，每打68美元，2打迷你纸杯蛋糕，每打80美元，还有6打迷你奶酪蛋糕，每打55美元。总共花了多少钱？

卡洛斯正在种植一棵柠檬树。这棵树将花费90美元来种植。每年它将长出7个柠檬，他可以以每个1.5美元的价格出售。每年浇水和喂养这棵树的费用是3美元。他需要多少年才能靠这棵柠檬树挣钱？

梅兰妮是一个挨家挨户的推销员。她把三分之一的吸尘器卖给了绿房子，又把两个卖给了红房子，剩下的一半卖给了橙房子。如果梅兰妮还剩下5个吸尘器，她一开始有多少个？

在一个有20名学生的舞蹈班里，有20%的学生报名参加了现代舞，剩下的25%报名参加了爵士舞，其余的报名参加了嘻哈舞。整个学生中报考嘻哈舞的比例是多少？

一个商人想在2个购买计划中做出选择：价值5000美元的珠宝或价值8000美元的电子小玩意。他的财务顾问推测，珠宝市场将上涨2.5%，而电子小工具市场将在同一个月内上涨1.2%。如果该商人希望在本月底通过选择实现利润最大化，这将是多少的利润？

两列火车同时离开圣拉斐尔。它们开始向西行驶，都行驶了80英里。第二天，它们向北行驶，行驶了150英里。两天内，每辆火车行驶的距离是多少？

吉尔教书的工资是每小时20美元，做拉拉队的教练是30美元。如果她一年工作50周，每周做35小时教师，每周做15小时教练，她的年薪是多少？

克莱尔每天早上做一个3个鸡蛋的煎蛋卷作为早餐。 她在4周内会吃多少打鸡蛋？

玛丽莎正在徒步走一条12英里的小路。她花了1小时走完前4英里，然后又花了1小时走完后2英里。如果她希望她的平均速度是每小时4英里，那么她需要用什么速度（以每小时英里为单位）来走完剩下的路程？"""
train = translated_gsm8k_train.split("\n\n")

with open(config["GSM8K_path"] + "train.jsonl") as f:
    samples = list(map(json.loads, f.readlines()))

train = [{"question": q, "answer": s['answer']}
         for q, s in zip(train, samples[:len(train)])]

with open('data/GSM8K/train_cn.jsonl', 'w', encoding='utf-8') as f:
    for entry in train:
        json.dump(entry, f, ensure_ascii=False)
        f.write('\n')
