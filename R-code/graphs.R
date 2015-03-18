library(ggplot2)

SSE = data.frame(SSE = numeric(0), expr = character(0))
data = read.csv('../../results/boston_test.csv')
SSE = rbind(SSE, data.frame(SSE = (data$Y0 - data$mu0)^2, expr = 'test'))
graph(SSE, 'boston')

graph = function(data, name){
  ggplot(subset(data,  TRUE), aes(x = expr, y = SSE, fill  = expr)) + 
    stat_summary(fun.y = "mean", geom = "bar", position = position_dodge() ) + 
    stat_summary(fun.data = "mean_cl_boot", geom="linerange", colour="black", position=position_dodge(.9)) +
    theme_bw() +
    theme(legend.direction = "vertical", legend.position = "right", legend.box = "vertical", 
          axis.line = element_line(colour = "black"),
          panel.grid.major=element_blank(), 
          panel.grid.minor=element_blank(),      
          panel.border = element_blank(),
          text=element_text(size=10),
          legend.key = element_blank()
    ) +
    guides(fill = guide_legend(keywidth = 0.5, keyheight = 0.5))
  ggsave(file=paste("../../graphs/test", name, ".pdf", sep = ""),  width=7, height=4, units = "cm")    
  
}