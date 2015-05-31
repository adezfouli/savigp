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

output_path = "../../SAVIGP_paper/nips2015/figures/raw/"

# boston data
name= 'boston'
data = read.csv('../../graph_data/boston_SSE_data.csv')
p1 = draw_boxplot_models(data, "SSE", "None")

data = read.csv('../../graph_data/boston_NLPD_data.csv')
p2 = draw_boxplot_models(data, "NLPD", "right")

g = arrangeGrob(p1, p2, ncol=2,  widths=c(9/20, 11/20))
ggsave(file=paste(output_path, name, ".pdf", sep = ""),  width=16, height=5, units = "cm" , device=cairo_pdf, g)      

# abalone data
name= 'alablone'
data = read.csv('../../graph_data/abalone_SSE_data.csv')
p1 = draw_boxplot_models(data, "SSE", "None")

data = read.csv('../../graph_data/abalone_NLPD_data.csv')
p2 = draw_boxplot_models(data, "NLPD", "right")

g = arrangeGrob(p1, p2, ncol=2,  widths=c(9/20, 11/20))
ggsave(file=paste(output_path, name, ".pdf", sep = ""),  width=16, height=5, units = "cm" , device=cairo_pdf, g)      

# creep data
name = "creep"
data = read.csv('../../graph_data/creep_SSE_data.csv')
p1 = draw_boxplot_models(data, "SSE", "None")

data = read.csv('../../graph_data/creep_NLPD_data.csv')
p2 = draw_boxplot_models(data, "NLPD", "right")

g = arrangeGrob(p1, p2, ncol=2,  widths=c(9/20, 11/20))
ggsave(file=paste(output_path, name, ".pdf", sep = ""),  width=16, height=5, units = "cm" , device=cairo_pdf, g)      

# mining data
data = read.csv('../../graph_data/mining_intensity_data.csv')
data$model = substr(data$model_sp,0, 4)
data$sp = substr(data$model_sp,6, 8)
draw_intensity(data, "intensity", "mining_intensity", "../../SAVIGP_paper/nips2015/figures/raw/")

# wisc data ####
name = "wisc"
data = read.csv('../../graph_data/wisc_ER_data.csv')
p1 = draw_bar_models(data, "ER", "None")

data = read.csv('../../graph_data/wisc_NLPD_data.csv')
p2 = draw_boxplot_models(data, "NLPD",  "right")

g = arrangeGrob(p1, p2, ncol=2,  widths=c(9/20, 11/20))
ggsave(file=paste(output_path, name, ".pdf", sep = ""),  width=16, height=5, units = "cm" , device=cairo_pdf, g)      

#usps data
name = "usps"
data = read.csv('../../graph_data/usps_ER_data.csv')
p1 = draw_bar_models(data, "ER", "None")

data = read.csv('../../graph_data/usps_NLPD_data.csv')
p2 = draw_boxplot_models(data, "NLPD", "right")

g = arrangeGrob(p1, p2, ncol=2,  widths=c(9/20, 11/20))
ggsave(file=paste(output_path, name, ".pdf", sep = ""),  width=16, height=5, units = "cm" , device=cairo_pdf, g)      

#mnist data
name = "mnist"
data = read.csv('../../graph_data/mnist_ER_data.csv')
p1 = draw_bar_models(data, "ER", "None")

data = read.csv('../../graph_data/mnist_NLPD_data.csv')
p2 = draw_boxplot_models(data, "NLPD", "right")

g = arrangeGrob(p1, p2, ncol=2,  widths=c(9/20, 11/20))
ggsave(file=paste(output_path, name, ".pdf", sep = ""),  width=16, height=5, units = "cm" , device=cairo_pdf, g)      

# heloper funs ####
draw_bar_models <- function(data, y_lab, leg_pos){
  data$X = NULL
  data = melt(data)
  data$model = substr(data$variable,0, 4)
  data$sp = substr(data$variable,6, 15)
  
  ggplot(data, aes(x="", y = value, fill = sp)) + 
    stat_summary(fun.y = "mean", geom = "bar", position = position_dodge() ) + 
    stat_summary(fun.data = mean_cl_normal, geom="linerange", colour="black", position=position_dodge(.9)) +
    theme_bw() + 
    scale_fill_brewer(palette="Set1") + 
    
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
          legend.title=element_blank(),
          axis.ticks.x = element_blank(),
          legend.title=element_blank(),
          axis.title.x=element_blank()
          
          
    ) +
    facet_wrap(~model)+ 
    
    guides(fill = guide_legend(keywidth = 0.5, keyheight = 0.5))
}  
  

draw_boxplot_models <- function(data, y_lab, leg_pos){
  data$X = NULL
  data = melt(data)
  data$model = substr(data$variable,0, 4)
  data$sp = substr(data$variable,6, 15)
  
  
y_max = max(by(data, data[, c('model', 'sp')], function(x){quantile(x$value, 0.975)}))
y_min = min(by(data, data[, c('model', 'sp')], function(x){quantile(x$value, 0.025)}))



p = ggplot(data, aes(x='', y = value, colour = sp)) + 
  geom_boxplot(width=1, 
               position=position_dodge(1),
               outlier.shape = 16, outlier.size = .2) + 
  coord_cartesian(y = c(y_min, y_max)) +
  theme_bw() + 
  scale_colour_brewer(palette="Set1") +
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
        legend.title=element_blank(),
        axis.title.x=element_blank()
  ) +
  facet_wrap(~model)+ 
  
guides(fill = guide_legend(keywidth = 0.5, keyheight = 0.5))
p
}

draw_intensity <- function(data, y_lab, name, output_path){
  ggplot(data, aes(x=x, y = m, colour = sp)) + 
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
  
  ggsave(file=paste(output_path, name, ".pdf", sep = ""),  width=10, height=5, units = "cm", device=cairo_pdf)      
}

