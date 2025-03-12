# Loss
## CrossEntropyLoss
- [torch.nn.CrossEntropyLoss](https://pytorch.org/docs/2.5/generated/torch.nn.CrossEntropyLoss.html#crossentropyloss)
- 适用于C类分类问题，可选变量weight为1D张量，对于不平衡训练集尤其有用。
- 带类别索引表达式 $l(x,y)=L=\{ l_1,\cdots, l_N\}^T, \quad l_n=-w_{y_n}\log\frac{\exp(x_{n,y_n})}{\sum_{c=1}^C\exp(x_{n,c})}.1\{ y_n \neq ignore\_index\}$
- 其中$x$为输入，$y$为目标，$w$为权重，$C$为类别数，$N$为batch数
- $$l(x,y)=\begin{cases}\sum_{n=1}^N\frac{1}{\sum_{n=1}^Nw_{y_n}.1\{y_n \neq ignore\_index\}}l_n , \quad  &if \quad  reduction= 'mean'\\ \sum_{n=1}^Nl_n, \quad  &if \quad  reduction= 'sum'\end{cases}$$
- 每个类别的概率表达式: $l(x,y)=L=\{ l_1,\cdots, l_N\}^T, \quad l_n=-\sum_{c=1}^C w_c\log\frac{\exp(x_{n,c})}{\sum_{c=1}^C\exp(x_{n,i})}.y_{n,c}$
- $$l(x,y)=\begin{cases}\frac{\sum_{n=1}^Nl_n}{N} , \quad  &if \quad  reduction= 'mean'\\ \sum_{n=1}^Nl_n, \quad  &if \quad  reduction= 'sum'\end{cases}$$
## MSELoss
- [torch.nn.MSELoss](https://pytorch.org/docs/2.5/generated/torch.nn.MSELoss.html#torch.nn.MSELoss)
- $$l(x,y)=L=\{ l_1m \cdots, l_N\}^T, \quad l_n = (x_n - y_n)^2,$$
- $$l(x,y)=\begin{cases}mean(L) , \quad  &if \quad  reduction= 'mean'\\ sum(L), \quad  &if \quad  reduction= 'sum'\end{cases}$$
## DiceLoss
## BCELoss
## FocalLoss



