library(languageR)
library(ggplot2)
library(plotrix)
library(plyr)
library(party)
library(gridExtra)
library(Hmisc)
library(extrafont)
library(scales)
library(reshape)
library(pbkrtest)
library(nloptr)
library(optimx)
library(data.table)
library(extrafont)
loadfonts()


SP_name = "SF"

w = 15
h = 5
output_path = "../../SAVIGP_paper/nips2015/figures/raw/"

# boston data
name= 'boston'
data = read.csv('../../graph_data/boston_SSE_data.csv')
p1 = draw_boxplot_models(data, "SSE", "None")

data = read.csv('../../graph_data/boston_NLPD_data.csv')
p2 = draw_boxplot_models(data, "NLPD", "right")

g = arrangeGrob(p1, p2, ncol=2,  widths=c(9/20, 11/20))
ggsave(file=paste(output_path, name, ".pdf", sep = ""),  width=w, height=h, units = "cm" , device=cairo_pdf, g)      

# abalone data
name= 'abalone'
data = read.csv('../../graph_data/abalone_SSE_data.csv')
p1 = draw_boxplot_models(data, "SSE", "None")

data = read.csv('../../graph_data/abalone_NLPD_data.csv')
p2 = draw_boxplot_models(data, "NLPD", "right")

g = arrangeGrob(p1, p2, ncol=2,  widths=c(9/20, 11/20))
ggsave(file=paste(output_path, name, ".pdf", sep = ""),  width=w, height=h, units = "cm" , device=cairo_pdf, g)      

# abalone data timing
name= 'abalone'
data = read.csv('../../graph_data/abalone_timing_SSE_data.csv')
data$X = NULL
g = draw_bar_timing(data, "", "")
ggsave(file=paste(output_path, "abalone_timing", ".pdf", sep = ""),  width=w, height=h, units = "cm" , device=cairo_pdf, g)      

# creep data
name = "creep"
data = read.csv('../../graph_data/creep_SSE_data.csv')
p1 = draw_boxplot_models(data, "SSE", "None")

data = read.csv('../../graph_data/creep_NLPD_data.csv')
p2 = draw_boxplot_models(data, "NLPD", "right")

g = arrangeGrob(p1, p2, ncol=2,  widths=c(9/20, 11/20))
ggsave(file=paste(output_path, name, ".pdf", sep = ""),  width=w, height=h, units = "cm" , device=cairo_pdf, g)      


# mining data
name = 'mining'
data = read.csv('../../graph_data/mining_intensity_data.csv')
data$model = toupper(substr(data$model_sp,0, 4))
data = rename_model(data)
data$sp = paste(SP_name, "=", substr(data$model_sp,6, 8))
p2 = draw_intensity(data, "intensity")

data = read.csv('../../graph_data/mining_true_y_data.csv')
p1 = draw_mining_data(data)
g = arrangeGrob(p1, p2, ncol=2,  widths=c(8/20, 12/20))
ggsave(file=paste(output_path, name, ".pdf", sep = ""),  width=w, height=h, units = "cm" , device=cairo_pdf, g)      



# wisc data ####
name = "wisc"
data = read.csv('../../graph_data/wisc_ER_data.csv')
p1 = draw_bar_models(data, "error rate", "None")

data = read.csv('../../graph_data/wisc_NLPD_data.csv')
p2 = draw_boxplot_models(data, "NLP",  "right")

g = arrangeGrob(p1, p2, ncol=2,  widths=c(9/20, 11/20))
ggsave(file=paste(output_path, name, ".pdf", sep = ""),  width=w, height=h, units = "cm" , device=cairo_pdf, g)      


#usps data
name = "usps"
data = read.csv('../../graph_data/usps_ER_data.csv')
p1 = draw_bar_models(data, "error rate", "None")

data = read.csv('../../graph_data/usps_NLPD_data.csv')
p2 = draw_boxplot_models(data, "NLP", "right")

g = arrangeGrob(p1, p2, ncol=2,  widths=c(9/20, 11/20))
ggsave(file=paste(output_path, name, ".pdf", sep = ""),  width=w, height=h, units = "cm" , device=cairo_pdf, g)      

# abalone data timing
name= 'mnist'
data = read.csv('../../graph_data/mnist_timing_SSE_data.csv')
data$X = NULL
g = draw_bar_timing(data, "", "")
ggsave(file=paste(output_path, "abalone_timing", ".pdf", sep = ""),  width=w, height=h, units = "cm" , device=cairo_pdf, g)      



#mnist data
name = "mnist_sarcos"
data = read.csv('../../graph_data/mnist_ER_data.csv')
p1 = draw_bar_models_with_X(data, "error rate", "None")

data = read.csv('../../graph_data/mnist_NLPD_data.csv')
p2 = draw_boxplot_models_with_X(data, "NLP", "None")


data = read.csv('../../graph_data/sarcos_MSSE_data.csv')
p3 = draw_joints(data)

g = arrangeGrob(p1, p2, p3, ncol=3,  widths=c(11/30, 11/30, 8/30))
ggsave(file=paste(output_path, name, ".pdf", sep = ""),  width=w, height=h, units = "cm" , device=cairo_pdf, g)      


# heloper funs ####
rename_model <- function(data){
  data$model[data$model == 'MIX1'] = "MoG1"
  data$model[data$model == 'MIX2'] = "MoG2"
  data$model[data$model == 'FULL'] = "FG"
  data
}

draw_bar_timing <- function(data, y_lab, leg_pos){
  data = melt(data)
  data$time = as.integer(substr(data$variable, 9, 15))
  
  plot_data = aggregate(value ~ time, FUN = mean, data =data)
  
  p = ggplot(plot_data, aes(x=time / 1000, y = value)) + 
    geom_point() + 
    geom_line() + 
    coord_cartesian(ylim = c(y_min - abs(y_min) * 0.1 , y_max + abs(y_max) * 0.1)) +
    theme_bw() + 
    scale_colour_brewer( palette="Set1") +
    ylab("SSE") +
    xlab("time (s)")
    theme(legend.direction = "vertical", legend.position = leg_pos, legend.box = "vertical", 
          axis.line = element_line(colour = "black"),
          panel.grid.major=element_blank(), 
          panel.grid.minor=element_blank(),      
          panel.border = element_blank(),
          panel.margin = unit(.4, "lines"),
          text=element_text(family="Arial", size=10),
          legend.key = element_blank(),
          strip.background = element_rect(colour = "white", fill = "white",
                                          size = 0.5, linetype = "solid")
    ) 
    
    guides(fill = guide_legend(keywidth = 0.5, keyheight = 0.5))
    p
    
}  


draw_bar_models <- function(data, y_lab, leg_pos){
  data$X = NULL
  data = melt(data)
  data$model = toupper(substr(data$variable,0, 4))
  data = rename_model(data)
  data$sp = substr(data$variable,6, 15)
  
  ggplot(data, aes(x="", y = value, fill = sp)) + 
    stat_summary(fun.y = "mean", geom = "bar", position = position_dodge() ) + 
    stat_summary(fun.data = mean_cl_normal, geom="linerange", colour="black", position=position_dodge(.9)) +
    theme_bw() + 
    scale_fill_brewer(name=SP_name, palette="Set1") + 
    
    xlab('') +
    ylab(y_lab) +
    theme(legend.direction = "vertical", legend.position = leg_pos, legend.box = "vertical", 
          axis.line = element_line(colour = "black"),
          panel.grid.major=element_blank(), 
          panel.grid.minor=element_blank(),      
          panel.border = element_blank(),
          panel.margin = unit(.4, "lines"),
          text=element_text(family="Arial", size=10),
          legend.key = element_blank(),
          strip.background = element_rect(colour = "white", fill = "white",
                                          size = 0.5, linetype = "solid"),
          axis.ticks.x = element_blank(),
          axis.title.x=element_blank()
          
          
    ) +
    facet_wrap(~model)+ 
    
    guides(fill = guide_legend(keywidth = 0.5, keyheight = 0.5))
}  
  
draw_bar_models_with_X <- function(data, y_lab, leg_pos){
  data$X = NULL
  data = melt(data)
  data$model = toupper(substr(data$variable,0, 4))
  data = rename_model(data)
  data$sp = substr(data$variable,6, 15)
  
  ggplot(data, aes(x=sp, y = value, fill = sp)) + 
    stat_summary(fun.y = "mean", geom = "bar", position = position_dodge() ) + 
    stat_summary(fun.data = mean_cl_normal, geom="linerange", colour="black", position=position_dodge(.9)) +
    theme_bw() + 
    scale_fill_brewer(name=SP_name, palette="Set1") + 
    xlab(SP_name) +
    ylab(y_lab) +
    theme(legend.direction = "vertical", legend.position = leg_pos, legend.box = "vertical", 
          axis.line = element_line(colour = "black"),
          panel.grid.major=element_blank(), 
          panel.grid.minor=element_blank(),      
          panel.border = element_blank(),
          panel.margin = unit(.4, "lines"),
          text=element_text(family="Arial", size=10),
          legend.key = element_blank(),
          strip.background = element_rect(colour = "white", fill = "white",
                                          size = 0.5, linetype = "solid")
          
          
          
    ) +
    facet_wrap(~model)+ 
    
    guides(fill = guide_legend(keywidth = 0.5, keyheight = 0.5))
}  



draw_boxplot_models_with_X <- function(data, y_lab, leg_pos){
  data$X = NULL
  data = melt(data)
  data$model = toupper(substr(data$variable,0, 4))
  data$sp = substr(data$variable,6, 15)
  data = rename_model(data)
  #y_max = max(by(data, data[, c('model', 'sp')], function(x){quantile(x$value, 0.975)}))
  #y_min = min(data$value)
  y_max = max(by(data, data[, c('model', 'sp')], function(x){boxplot.stats(x$value)$stats[c(5)]}))
  y_min = min(by(data, data[, c('model', 'sp')], function(x){boxplot.stats(x$value)$stats[c(1)]}))
  
  
  p = ggplot(data, aes(x=sp, y = value, colour = sp)) + 
  geom_boxplot(width=1, 
               position=position_dodge(1),
               outlier.shape = NA) + 
  coord_cartesian(ylim = c(y_min - abs(y_min) * 0.1 , y_max + abs(y_max) * 0.1)) +
  theme_bw() + 
  scale_colour_brewer(name=SP_name, palette="Set1") +
  xlab(SP_name) +
  ylab(y_lab) +
  theme(legend.direction = "vertical", legend.position = leg_pos, legend.box = "vertical", 
        axis.line = element_line(colour = "black"),
        panel.grid.major=element_blank(), 
        panel.grid.minor=element_blank(),      
        panel.border = element_blank(),
        panel.margin = unit(.4, "lines"),
        text=element_text(family="Arial", size=10),
        legend.key = element_blank(),
        strip.background = element_rect(colour = "white", fill = "white",
                                        size = 0.5, linetype = "solid")
  ) +
  facet_wrap(~model)+ 
  
guides(fill = guide_legend(keywidth = 0.5, keyheight = 0.5))
p
}

draw_intensity <- function(data, y_lab){
  p = ggplot(data, aes(x=x, y = m, colour = sp)) + 
    geom_line() +
    geom_ribbon(aes(x=x, ymin= m - 2 * sqrt(v), ymax=m + 2 * sqrt(v)), fill="grey", alpha=.4, colour =NA) +  
    scale_colour_brewer(palette="Set1") +
    xlab('') +
    ylab(y_lab) +
    theme_bw() + 
    
    theme(legend.direction = "vertical", legend.position = "none", legend.box = "vertical", 
          axis.line = element_line(colour = "black"),
          panel.grid.minor=element_blank(),      
          panel.border = element_blank(),
          panel.margin = unit(.4, "lines"),
          text=element_text(family="Arial", size=10),
          legend.key = element_blank(),
          strip.background = element_rect(colour = "white", fill = "white",
                                          size = 0.5, linetype = "solid"),
          axis.ticks.x = element_blank(),
          legend.title=element_blank(),
          axis.text.x = element_text(angle = 90, hjust = 1)
    ) +
    facet_grid(model ~ sp)+ 
    guides(fill = guide_legend(keywidth = 0.5, keyheight = 0.5))
    p  
}

draw_mining_data <- function(data){
  data$X = NULL
  p = ggplot(data, aes(x=x, y = y)) + 
    stat_summary(fun.y = "mean", geom = "line", position = position_dodge()) + 
    
    
    theme_bw() + 
    scale_colour_brewer(palette="Set1") +
    xlab('time') +
    ylab('event counts') +
    theme(legend.direction = "vertical", legend.position = "none", legend.box = "vertical", 
          axis.line = element_line(colour = "black"),
          panel.grid.major=element_blank(), 
          panel.grid.minor=element_blank(),      
          panel.border = element_blank(),
          panel.margin = unit(.4, "lines"),
          text=element_text(family="Arial", size=10),
          legend.key = element_blank(),
          strip.background = element_rect(colour = "white", fill = "white",
                                          size = 0.5, linetype = "solid")
    ) +
    guides(fill = guide_legend(keywidth = 0.5, keyheight = 0.5))
  p
}

draw_joints <- function(data){
  data$X = NULL
  data = melt(data)
  data$joint = factor(as.numeric(substr(data$variable,11, 14)) + 1)
  data$name = paste(SP_name, "=", "0.04")
  p =   ggplot(data, aes(x=joint, y = value)) + 
    stat_summary(fun.y = "mean", geom = "bar", fill="gray", colour = "black",position = position_dodge() ) + 
    theme_bw() + 
    
    xlab("output") +
    ylab("SMSE") +
    theme(legend.direction = "vertical", legend.position = "none", legend.box = "vertical", 
          axis.line = element_line(colour = "black"),
          panel.grid.major=element_blank(), 
          panel.grid.minor=element_blank(),      
          panel.border = element_blank(),
          panel.margin = unit(.4, "lines"),
          text=element_text(family="Arial", size=10),
          legend.key = element_blank(),
          strip.background = element_rect(colour = "white", fill = "white",
                                          size = 0.5, linetype = "solid")
          
          
    ) +
    facet_grid(. ~ name)+ 
    
    
    guides(fill = guide_legend(keywidth = 0.5, keyheight = 0.5))
  p
}



