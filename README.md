# HMBS 写真预设库（FLUX）  

### 这个节点可以直接生成写真embedding，可以用于flux类模型。在有些你懒得自己些提示词的情况，可以直接用这个节点进行，测试，调试，或者演示。  



.  
.  
.  


---
## 安装方法

适用于大多数torch版本，没有用到其他库，大多数情况下有pytorch应该能直接用

将代码下载到 custom_nodes 文件夹下

将 HMBS_sty 文件解压到 models 文件夹下  

百度链接: [HMBS_sty](https://pan.baidu.com/s/1tbeTiORncLoj7NwXpqv8oA?pwd=HMBS "HMBS_sty 下载")

  
具体预设写真可以参考

-   佳丽.md  
-   气质写真.md  
-   特效写真.md  
-   道具写真.md  


.  
.  
.  

---

## 使用方式

代替clip encode 节点，直接输出embedding，可以参考 example/simple_workflow.json  
同时也适用于 nunchaku 加速节点  
gguf应该也没问题，但没有测试

.  
.  
.  

---
## 样图
![image](source/2025-05-14-00-22-38-2747/174817167087783.jpg)![image](source/2025-05-23-16-11-31-129/174817281286157.jpg)  
![image](source/2025-05-22-23-26-34-1259/174817128693512.jpg)![image](source/2025-05-24-17-39-38-6591/174817085776652.jpg)   

.  
.  
.  

---

最后呼吁某位连锁照相馆老板把用各种手段逼走的员工的离职补偿还给他们
