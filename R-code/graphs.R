library(ggplot2)

SSE = data.frame(SSE = numeric(0), expr = character(0))
path = '../../results/botson_WQXVO9/boston'
data_gp = read.csv(paste(path, '_gp_test.csv', sep=''))
data_savigp = read.csv(paste(path, '_savigp_test.csv', sep=''))
data_train = read.csv(paste(path, '_gp_train.csv', sep=''))

SSE = rbind(SSE, data.frame(SSE = (data_gp$Y0 - data_gp$mu0)^2 / mean((mean(data_train$Y0)-data_gp$Y0)^2), expr = 'gp'))
SSE = rbind(SSE, data.frame(SSE = (data_savigp$Y0 - data_savigp$mu0)^2 / mean((mean(data_train$Y0)-data_savigp$Y0)^2), expr = 'savigp'))


graph_bar(SSE, 'boston')

graph_bar = function(data, name){
  ggplot(subset(data,  TRUE), aes(x = expr, y = SSE, fill  = expr)) + 
    geom_boxplot() + 
#    stat_summary(fun.y = "mean", geom = "bar", position = position_dodge() ) + 
#    stat_summary(fun.data = "mean_cl_boot", geom="linerange", colour="black", position=position_dodge(.9)) +
    theme_bw() +
    ylim(c(0, 0.4)) + 
    theme(legend.direction = "vertical", legend.position = "right", legend.box = "vertical", 
          axis.line = element_line(colour = "black"),
          panel.grid.major=element_blank(), 
          panel.grid.minor=element_blank(),      
          panel.border = element_blank(),
          text=element_text(size=10),
          legend.key = element_blank()
    ) +
    guides(fill = guide_legend(keywidth = 0.5, keyheight = 0.5))
  ggsave(file=paste("../../graphs/test", name, ".pdf", sep = ""),  width=10, height=7, units = "cm")    
  
}

