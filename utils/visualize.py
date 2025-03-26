"""
用于学术论文的可视化工具
三类数据可视化图：
1. 趋势(trend)：一种变化模式
    1.1 折线图(Line chat) plot_line
2. 关系(relation)：直观反映数据中变量之间的关系
    2.1 条形图(Bar chart) plot_bar
    2.2 热力图(Heatmap) plot_heatmap
    2.3 散点图(Scatter plot) plot_scatter
    2.4 回归线散点图(regression line) plot_linreg
    2.5 多重回归线分类散点图 plot_linreg
    2.6 分类散点图(Categorical scatter plots) plot_cat_scatter
    2.7 关系组图 plot_rel
3. 分布(distribution)：直观反映数据的分布情况
    3.1 直方图(Hist chart) plot_hist
    3.2 核密度估计图(KDE chart) plot_kde
    3.3 箱线图(Box plot) plot_box
    3.4 小提琴图(Violin plot) plot_violin
    3.5 分类组图 plot_cat
    3.6 联合分布图(Joint plot) plot_joint
    3.7 成对关系图(Pair plot) plot_pair
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def set_sci_style():
    """
    统一样式，设置学术论文绘图标准：字体、背景、文本
    """
    # Set scientific style
    plt.style.use('seaborn-v0_8-paper')
    sns.set_context("paper", font_scale=1.5)
    sns.set_style("whitegrid", {
        "grid.linestyle": "--",
        "grid.alpha": 0.7,
        'font.family': 'serif',
        'font.serif': 'Times New Roman',
        'mathtext.fontset': 'stix',
        'font.size': 12,
        'axes.linewidth': 1.5,        # 加粗坐标轴线宽
        'axes.edgecolor': 'black',     # 设置坐标轴颜色为黑色
    })

"""
趋势图：
1. 折线图(Line chart)：显示一段时间内的趋势
-------------------------------------------------------------------------------
"""

def plot_line(df: pd.DataFrame, xlabel, ylabel, save_path, figsize=(12, 6), title=''):
    """_summary_
    绘制折线图, x轴必须规范化为 Number or Date
    Args:
        df (pd.DataFrame): index 为 x 轴数据，每一列为一个 y 轴数据
        xlabel (_type_): _description_
        ylabel (_type_): _description_
        save_path (_type_): _description_
        figsize (tuple, optional): _description_. Defaults to (12, 6).
        title (str, optional): _description_. Defaults to ''.
    """
    set_sci_style()
    # Create figure
    plt.figure(figsize=figsize)
    
    # Plot multiple lines
    sns.lineplot(data=df)
    
    # Customize plot
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if title:
        plt.title(title)
    
    # Add grid and legend
    plt.legend(loc='best')
    
    # Adjust layout
    plt.tight_layout()
    
    # Save plot
    plt.savefig(save_path+'_line.pdf', format='pdf', bbox_inches='tight', dpi=300)
    plt.show()

"""
关系图：
1. 条形图(Bar chart)：比较不同组别对应的数量，误差线提供估计值附近的不确定性
2. 热力图(Heatmap)：查看所有特征两两之间的关系，为数字矩阵创建色彩矩阵
3. 散点图(Scatter plot)：反映两个连续变量之间的关系；若用颜色编码，还可显示与第三个分类变量的关系
4. 回归线散点图(regression line)：反映两个连续变量的线性关系
5. 多重回归线分类散点图：反映两个连续变量在第三个分类变量内的线性关系
6. 分类散点图(Categorical scatter plots)：展示连续变量与分类变量之间的关系
    6.1 分布散点图(strip): 其中一个变量是分类变量的散点图
    6.2 分布密度散点图(swarm): 对点沿着分类轴进行调整，使每个点互不重叠，不适合大量观测的可视化
-------------------------------------------------------------------------------
"""
    
def plot_bar(df: pd.DataFrame, x, y, xlabel, ylabel, save_path, figsize=(12, 8), title='', value=None, hue=None, color='skyblue', palette='Paired'):
    """_summary_
    绘制分类特征与连续特征的条形图，包含误差线（默认为均值、标准差）
    Args:
        df (pd.DataFrame): 包含连续特征和分类特征
        x (_type_): x 轴所选列（分类/连续特征）
        y (_type_): y 轴所选列（连续/分类特征）
        xlabel (_type_): _description_
        ylabel (_type_): _description_
        save_path (_type_): _description_
        figsize (tuple, optional): _description_. Defaults to (12, 8).
        title (str, optional): _description_. Defaults to ''.
        value (_type_, optional): 是否在条形边缘显示数值，0是竖条，1是横条. Defaults to None.
        hue (_type_, optional): 根据该分类特征进行颜色编码. Defaults to None.
        color (str, optional): 无分类特征时的颜色. Defaults to 'skyblue'.
        palette (str, optional): 分类特征时的调色板. Defaults to 'Paired'.
    """
    set_sci_style()
    # Create figure
    plt.figure(figsize=figsize)
    
    # ax = sns.barplot(x=x, y=y, hue=y)
    ax = None
    if hue is None:
        ax = sns.barplot(x=x, y=y, data=df, color=color)
    else:
        ax = sns.barplot(x=x, y=y, data=df, hue=hue, palette=palette)
        
    if value == 0:
        # 添加横向数值标签
        for container in ax.containers:
            ax.bar_label(container, fmt='%.2f', label_type='edge')
    elif value == 1:
        # 添加竖向数值标签
        for container in ax.containers:
            ax.bar_label(container, fmt='%.2f', label_type='edge')
    
    # Customize plot
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if title:
        plt.title(title)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save plot
    plt.savefig(save_path+'_bar.pdf', format='pdf', bbox_inches='tight', dpi=300)
    plt.show()

def plot_heatmap(df: pd.DataFrame, xlabel, ylabel, save_path, figsize=(12, 8), title=''):
    """_summary_
    绘制热力图，查看所有特征两两之间的关系，为数字矩阵创建色彩矩阵
    Args:
        df (_type_): index 和 columns 是特征名，表项必须是连续值
        xlabel (_type_): _description_
        ylabel (_type_): _description_
        save_path (_type_): _description_
        figsize (tuple, optional): _description_. Defaults to (12, 8).
        title (str, optional): _description_. Defaults to ''.
    """
    set_sci_style()
    # Create figure
    plt.figure(figsize=figsize)
    
    # Create heatmap
    # sns.heatmap(df, annot=True, cmap='mako')
    sns.heatmap(df, mask=df.isnull(), annot=True, cmap='rocket')
    
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
        
    # Customize plot
    if title:
        plt.title(title)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save plot
    plt.savefig(save_path+'_heatmap.pdf', format='pdf', bbox_inches='tight', dpi=300)
    plt.show()

def plot_scatter(df: pd.DataFrame, xlabel, ylabel, save_path, figsize=(12, 8), title='', hue=None):
    """_summary_
    绘制散点图，反映两个连续变量之间的关系；若用颜色编码，还可显示与第三个分类变量的关系
    Args:
        df (pd.DataFrame): 列为特征，行为样本
        xlabel (_type_): 自变量
        ylabel (_type_): 因变量
        save_path (_type_): _description_
        figsize (tuple, optional): _description_. Defaults to (12, 8).
        title (str, optional): _description_. Defaults to ''.
        hue (_type_, optional): 第三个分类变量. Defaults to None.
    """
    set_sci_style()
    # Create figure
    plt.figure(figsize=figsize)
    
    #Plot
    sns.scatterplot(x=xlabel, y=ylabel, data=df, hue=hue, palette='deep')
    
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
        
    # Customize plot
    if title:
        plt.title(title)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save plot
    plt.savefig(save_path+'_scatter.pdf', format='pdf', bbox_inches='tight', dpi=300)
    plt.show()

def plot_linreg(df: pd.DataFrame, xlabel, ylabel, save_path, figsize=(12, 8), title='', hue=None, row=None, col=None):
    """_summary_
    绘制带有回归直线的散点图，反映两个连续变量之间的关系;
    若用颜色编码，还可显示与第三个分类变量的关系;
    绘制回归曲线组图，根据 row 和 col 横纵分组，hue 为第三个分类变量
    Args:
        df (pd.DataFrame): 列为特征，行为样本
        xlabel (_type_): 自变量
        ylabel (_type_): 因变量
        save_path (_type_): _description_
        figsize (tuple, optional): _description_. Defaults to (12, 8).
        title (str, optional): _description_. Defaults to ''.
        hue (_type_, optional): 第三个分类变量. Defaults to None.
        row (_type_, optional):
        col (_type_, optional):
    """
    set_sci_style()
    if hue is None:
        # 创建图形
        plt.figure(figsize=figsize)
        
        # 绘制回归图
        sns.regplot(x=xlabel, y=ylabel, data=df)
        
        # 设置标签
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        
        # 设置标题
        if title:
            plt.title(title)
        
        # 调整布局并保存
        plt.tight_layout()
        plt.savefig(save_path+'_linreg.pdf', format='pdf', bbox_inches='tight', dpi=300)
        plt.show()
    else:
        # 使用 lmplot 创建 FacetGrid 对象
        g = sns.lmplot(x=xlabel, y=ylabel, hue=hue, data=df, height=figsize[1], aspect=figsize[0]/figsize[1], palette='deep', col=col, row=row)
        
        if g._legend is not None:
            # 获取图例句柄和标签
            handles, labels = g._legend_data.values(), g._legend_data.keys()
            # 移除默认图例
            g._legend.remove()
            
            # 添加全局图例，位于图外
            g.figure.legend(handles=handles, labels=labels, title=hue, 
                        loc='upper right', bbox_to_anchor=(1.2, 1))
        
        # 设置标题
        if title:
            g.figure.suptitle(title, y=1.02)  # y 参数调整标题位置
        
        # 调整布局
        plt.tight_layout()
        
        # 保存图形
        g.savefig(save_path+'_linreg.pdf', format='pdf', bbox_inches='tight', dpi=300)
        plt.show()

def plot_cat_scatter(df: pd.DataFrame, xlabel, ylabel, save_path, figsize=(12, 8), title='', hue=None):
    """_summary_
    分类散点图：绘制分布散点图(strip)和分布密度散点图(swarm)两张图
    展示连续变量与分类变量之间的关系
    Args:
        df (pd.DataFrame): 列为特征，行为样本
        xlabel (_type_): 分类/连续特征列
        ylabel (_type_): 连续/分类特征列
        save_path (_type_): _description_
        figsize (tuple, optional): _description_. Defaults to (12, 8).
        title (str, optional): _description_. Defaults to ''.
        hue (_type_, optional): 第三个分类变量. Defaults to None.
    """
    set_sci_style()
    # Create figure
    plt.figure(figsize=figsize)
    
    #Plot
    if hue is None:
        sns.stripplot(x=xlabel, y=ylabel, data=df)
    else:
        sns.stripplot(x=xlabel, y=ylabel, data=df, hue=hue, palette='deep')
    
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
        
    # Customize plot
    if title:
        plt.title(title)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save plot
    plt.savefig(save_path+'_strip.pdf', format='pdf', bbox_inches='tight', dpi=300)
    plt.show()
    
    # Create figure
    plt.figure(figsize=figsize)
    
    #Plot
    if hue is None:
        sns.swarmplot(x=xlabel, y=ylabel, data=df)
    else:
        sns.swarmplot(x=xlabel, y=ylabel, data=df, hue=hue, palette='deep')
    
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
        
    # Customize plot
    if title:
        plt.title(title)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save plot
    plt.savefig(save_path+'_swarm.pdf', format='pdf', bbox_inches='tight', dpi=300)
    plt.show()

def plot_rel(df: pd.DataFrame, xlabel, ylabel, save_path, title='', hue=None, row=None, col=None, kind='scatter'):
    """_summary_
    绘制关系组图，可选：散点图、折线图。
    根据 row 和 col 横纵分组，hue 为第三个分类变量
    Args:
        df (pd.DataFrame): 列为特征，行为样本
        xlabel (_type_): 自变量
        ylabel (_type_): 因变量
        save_path (_type_): _description_
        figsize (tuple, optional): _description_. Defaults to (12, 8).
        title (str, optional): _description_. Defaults to ''.
        hue (_type_, optional): 第三个分类变量. Defaults to None.
        row (_type_, optional):
        col (_type_, optional):
        kind (str, optional): 图形类型，scatter 或 line. Defaults to 'scatter'.
    """
    set_sci_style()
    
    #Plot
    g = sns.relplot(x=xlabel, y=ylabel, data=df, hue=hue, palette='deep', col=col, row=row, kind=kind)
    
    # 调整图例位置，使其位于图外
    
    
    if g._legend is not None:
        # 获取图例句柄和标签
        handles, labels = g._legend_data.values(), g._legend_data.keys()
        # 移除默认图例
        g._legend.remove()
        
        # 添加全局图例，位于图外
        g.figure.legend(handles=handles, labels=labels, title=hue, 
                    loc='upper right', bbox_to_anchor=(1.2, 1))
    
    # 设置标题
    if title:
        g.figure.suptitle(title, y=1.02)  # y 参数调整标题位置
    
    # Adjust layout
    plt.tight_layout()
    
    # Save plot
    plt.savefig(save_path+'_rel.pdf', format='pdf', bbox_inches='tight', dpi=300)
    plt.show()
    

"""
分布图：
1. 直方图(Hist chart)：数据集的直方图分布
2. 核密度估计图(KDE chart)：核密度估计绘制数据集的分布
3. 箱线图(Box plot)：显示分类数据的分布情况，包括最大值、最小值、中位数、四分位数
4. 小提琴图(Violin plot)：箱线图+核密度估计。显示分类数据的分布情况，包括最大值、最小值、中位数、四分位数
5. 联合分布图(Joint plot)：显示两个变量之间的联合分布
6. 成对关系图(Pair plot)：显示数据集中所有特征两两之间的关系
-------------------------------------------------------------------------------
"""

def plot_hist(df: pd.DataFrame, xlabel, save_path, figsize=(12, 8), title='', hue=None, kde=False):
    """_summary_
    统计该列特征数值分布的直方图
    Args:
        df (pd.DataFrame): 列为特征，行为样本
        xlabel (_type_): 特征列
        save_path (_type_): _description_
        figsize (tuple, optional): _description_. Defaults to (12, 8).
        title (str, optional): _description_. Defaults to ''.
        hue (_type_, optional): 分类变量. Defaults to None.
        kde (bool, optional): 是否绘制 kde 曲线. Defaults to False.
    """
    set_sci_style()
    # Create figure
    plt.figure(figsize=figsize)
    
    #Plot
    sns.histplot(x=xlabel, data=df, hue=hue, palette='deep', kde=kde)
    
    plt.xlabel(xlabel)
        
    # Customize plot
    if title:
        plt.title(title)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save plot
    plt.savefig(save_path+'_hist.pdf', format='pdf', bbox_inches='tight', dpi=300)
    plt.show()

def plot_kde(df: pd.DataFrame, xlabel, save_path, ylabel=None, figsize=(12, 8), title='', hue=None):
    """_summary_
    统计该列特征数值分布的一维核密度估计图
    统计两列特征数值分布的二维核密度等高线图
    Args:
        df (pd.DataFrame): 列为特征，行为样本
        xlabel (_type_): 特征列
        save_path (_type_): _description_
        ylabel (_type_, optional): 特征列，若不为 None 则绘制等高线图. Defaults to None.
        figsize (tuple, optional): _description_. Defaults to (12, 8).
        title (str, optional): _description_. Defaults to ''.
        hue (_type_, optional): 分类变量. Defaults to None.
    """
    set_sci_style()
    # Create figure
    plt.figure(figsize=figsize)
    
    #Plot
    if  ylabel is None:
        sns.kdeplot(x=xlabel, data=df, hue=hue, palette='deep', fill=True)
    else:
        sns.kdeplot(x=xlabel, y=ylabel, data=df, hue=hue, palette='deep', fill=True, cbar=True)
        plt.ylabel(ylabel)
    
    plt.xlabel(xlabel)

        
    # Customize plot
    if title:
        plt.title(title)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save plot
    plt.savefig(save_path+'_kde.pdf', format='pdf', bbox_inches='tight', dpi=300)
    plt.show() 

def plot_box(df: pd.DataFrame, xlabel, ylabel, save_path, hue, figsize=(12, 8), title=''):
    """_summary_
    绘制箱线图，描述变量关于不同类别的分布情况
    框显示数据集的四分位数，中位数，最大值和最小值，同时可以显示离群点
    Args:
        df (pd.DataFrame): 列为特征，行为样本
        xlabel (_type_): 分类/连续特征列
        ylabel (_type_): 连续/分类特征列
        save_path (_type_): _description_
        hue (_type_): 无第三分类变量时与 xlabel 相同，有第三分类变量时为第三分类变量
        figsize (tuple, optional): _description_. Defaults to (12, 8).
        title (str, optional): _description_. Defaults to ''.
    """
    set_sci_style()
    # Create figure
    plt.figure(figsize=figsize)
    
    #Plot
    sns.boxplot(x=xlabel, y=ylabel, data=df, hue=hue, palette='deep')
    
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    
    # plt.grid(True, linestyle='--', alpha=0.7)
    
    # Customize plot
    if title:
        plt.title(title)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save plot
    plt.savefig(save_path+'_box.pdf', format='pdf', bbox_inches='tight', dpi=300)
    plt.show()

def plot_violin(df: pd.DataFrame, xlabel, ylabel, save_path, hue, figsize=(12, 8), title=''):
    """_summary_
    绘制小提琴图，通过箱线图得到数据对于分类变量的分位数，通过核密度估计得到数据的密度估计
    Args:
        df (pd.DataFrame): 列为特征，行为样本
        xlabel (_type_): 分类/连续特征列
        ylabel (_type_): 连续/分类特征列
        save_path (_type_): _description_
        hue (_type_): 无第三分类变量时与 xlabel 相同，有第三分类变量时为第三分类变量
        figsize (tuple, optional): _description_. Defaults to (12, 8).
        title (str, optional): _description_. Defaults to ''.
    """
    set_sci_style()
    # Create figure
    plt.figure(figsize=figsize)
    
    #Plot
    if xlabel == hue:
        sns.violinplot(x=xlabel, y=ylabel, data=df, hue=hue, palette='deep')
    else:
        sns.violinplot(x=xlabel, y=ylabel, data=df, hue=hue, palette='deep', split=True)
    
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    
    # Customize plot
    if title:
        plt.title(title)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save plot
    plt.savefig(save_path+'_violin.pdf', format='pdf', bbox_inches='tight', dpi=300)
    plt.show()

def plot_cat(df: pd.DataFrame, xlabel, ylabel, save_path, hue, title='', kind='violin', col=None, row=None):
    """_summary_
    分类变量的组图，展示分类变量与连续变量之间的关系
    Args:
        df (pd.DataFrame): 列为特征行为样本
        xlabel (_type_): 分类/连续特征列
        ylabel (_type_): 连续/分类特征列
        save_path (_type_): _description_
        hue (_type_): _description_
        title (str, optional): _description_. Defaults to ''.
        kind (str, optional): 'violin', 'box', 'bar', 'count', 'point'. Defaults to 'violin'.
        col (_type_, optional): _description_. Defaults to None.
        row (_type_, optional): _description_. Defaults to None.
    """
    set_sci_style()
    
    #Plot
    g = sns.catplot(x=xlabel, y=ylabel, data=df, hue=hue, palette='deep', kind=kind, col=col, row=row) 
    
    if g._legend is not None:
        # 调整图例位置，使其位于图外
        # 获取图例句柄和标签
        handles, labels = g._legend_data.values(), g._legend_data.keys()
        # 移除默认图例
        g._legend.remove()
        
        # 添加全局图例，位于图外
        g.figure.legend(handles=handles, labels=labels, title=hue, 
                    loc='upper right', bbox_to_anchor=(1.2, 1))
    
    # 设置标题
    if title:
        g.figure.suptitle(title, y=1.02)  # y 参数调整标题位置
    
    # Adjust layout
    plt.tight_layout()
    
    # Save plot
    plt.savefig(save_path+'_cat.pdf', format='pdf', bbox_inches='tight', dpi=300)
    plt.show()

def plot_joint(df: pd.DataFrame, xlabel, ylabel, save_path, hue=None, title=''):
    """_summary_
    显示两个连续变量之间的联合分布
    Args:
        df (pd.DataFrame): 列为特征，行为样本
        xlabel (_type_): 连续自变量
        ylabel (_type_): 连续因变量
        save_path (_type_): _description_
        hue (_type_, optional): 第三个分类变量. Defaults to None.
        title (str, optional): _description_. Defaults to ''.
    """
    set_sci_style()
    
    #Plot
    g = sns.jointplot(x=xlabel, y=ylabel, data=df, hue=hue, palette='deep')
    
    # 设置坐标轴标签
    g.set_axis_labels(xlabel, ylabel)
    
    
    # 设置标题
    if title:
        g.fig.suptitle(title, y=1.02)  # y 参数调整标题位置
    
    # Adjust layout
    plt.tight_layout()
    
    # Save plot
    plt.savefig(save_path+'_joint.pdf', format='pdf', bbox_inches='tight', dpi=300)
    plt.show()

def plot_pair(df: pd.DataFrame, save_path, hue=None, title=''):
    """_summary_
    显示数据集中所有特征两两之间的关系
    Args:
        df (pd.DataFrame): 列为特征，行为样本
        save_path (_type_): _description_
        hue (_type_, optional): 第三个分类变量. Defaults to None.
        title (str, optional): _description_. Defaults to ''.
    """
    set_sci_style()
    
    #Plot
    g = sns.pairplot(data=df, hue=hue, palette='deep')
    
    if g._legend is not None:
        # 调整图例位置，使其位于图外
        # 获取图例句柄和标签
        handles, labels = g._legend_data.values(), g._legend_data.keys()
        # 移除默认图例
        g._legend.remove()
        
        # 添加全局图例，位于图外
        g.figure.legend(handles=handles, labels=labels, title=hue, 
                    loc='upper right', bbox_to_anchor=(1.2, 1))
    
    # 设置标题
    if title:
        g.figure.suptitle(title, y=1.02)  # y 参数调整标题位置
    
    # Adjust layout
    plt.tight_layout()
    
    # Save plot
    plt.savefig(save_path+'_pair.pdf', format='pdf', bbox_inches='tight', dpi=300)
    plt.show()



if __name__ == "__main__":
    df = pd.DataFrame(sns.load_dataset('exercise'))
    print(df.head())
    plot_cat(df, 'time', 'pulse', 'img/exercise2', hue='kind')
    plot_cat(df, 'time', None, 'img/exercise3', hue='kind', kind='count', col='diet')
    