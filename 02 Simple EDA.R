




df %>% 
  group_by(label) %>% 
  summarise(Count = n())%>% 
  mutate(percent = prop.table(Count)*100)%>%
  ggplot(aes(reorder(label, -percent), percent), fill = Type)+
  geom_col(fill = c("grey", "light blue"))+
  geom_text(aes(label = sprintf("%.1f%%", percent)), hjust = 0.2, vjust = 2, size = 5)+ 
  theme_bw()+  
  xlab("Label") + ylab("Percent") 






# Companies by founded years
table(year(founded_at))

gg <- ggplot(data, aes(x=year(founded_at))) +
  geom_bar(stat="count" ,position="stack",alpha = 0.8)+
  geom_vline(aes(xintercept=mean(year(founded_at))),linetype="dashed")+
  ggtitle('Companies by founded years')+
  xlim(1990,2016)+
  xlab("Founded year") + ylab("Count")

ggplotly(gg, dynamicTicks = TRUE)


# Companies by founded status
table(company$status)

gg <- ggplot(company, aes(x=status)) +
  geom_bar(stat="count" ,position="stack",alpha = 0.8)+
  ggtitle('Companies by founded years')+
  xlab("Founded year") + ylab("Count")

ggplotly(gg, dynamicTicks = TRUE)

