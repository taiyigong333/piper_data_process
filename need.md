我现在使用使用项目/home/ubuntu/gcj/projects/data_collect/piper_with_pica采集了一些hdf5数据，数据在/home/ubuntu/gcj/projects/data_collect/data/corm_in_plate中。我现在希望将其转换为/home/ubuntu/gcj/projects/data_collect/data_processed/openpi中训练模型能够使用的lerobot data。

环境使用uv在/home/ubuntu/gcj/projects/data_collect/data_processed中配置。同时使用的lerobot 版本应该与pi项目的相同

数据处理的时候，需要根据/home/ubuntu/gcj/projects/data_collect/piper_with_pica/configs中对应的yaml来读取一些必要的值

同时处理数据的时候，机器人状态的频率高于画面的频率，需要做对齐。

同时使用config来控制一些参数，比如是提取关节角还是tcp。是使用绝对位姿还是相对位姿（同时config中也有一些说明性质的内容，比如是用于那个模型，使用数据的描述，但是这些值只用于对转换后的数据添加一个说明文件）。

同时图片的输入大小是1280x720的（也可能不是），需要添加一个可选的功能，可以将其变为设定的大小（比如224x224），具体做法是先保留长宽比的情况下，先变小，然后再使用padding将其补全到特定的大小

数据处理的代码放在/home/ubuntu/gcj/projects/data_collect/data_processed中，同时需要使用git进行相关的管理，并且善用子模块仓库（比如data_collect/data_processed/openpi就是一个子仓库）。调用的代码尽可能不要超过这个范围，需要使用的代码或者yaml直接在这里创建即可

需要放在远程仓库git@github.com:taiyigong333/piper_data_process.git中，密钥使用/home/ubuntu/.ssh/中的/home/ubuntu/.ssh/pris709