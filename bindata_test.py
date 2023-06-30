import numpy as np

#直接用np.fromfile读取二进制文件
bin = np.fromfile('/zhangpai25/dialog_dataset/qa_full.mmap', dtype=np.dtype('int32')).reshape(-1, 64+1024+4096)
print(bin.shape) #会卡住

# 直接加载文件并读取文件长度
with open('/zhangpai25/dialog_dataset/qa_full.mmap', 'r') as fid:
    nbytes = fid.seek(0, 2) #13417684992000
    flen = fid.tell() // np.dtype('int32').itemsize #3354421248000 = nbytes / 4
    print(nbytes, flen)

# 保存memmap文件？有什么注意事项？
bin = np.memmap('/zhangpai25/dialog_dataset/qa_full.mmap', dtype=np.dtype('int32'), shape=(flen // 64+1024+4096, 64+1024+4096))
print(type(bin))
print(bin.shape)