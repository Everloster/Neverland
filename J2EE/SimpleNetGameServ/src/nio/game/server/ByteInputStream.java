/************************************************************
* 文  件  名：ByteInputStream.java
* 功能模块：控制
* 功能描述：从一个Byte数组中读出一些基本数据类型
* 作          者：lee
************************************************************/
package nio.game.server;

import nio.game.utils.LOG;

/************************************************************
* 类          名：ByteInputStream
* 功能描述：从一个Byte数组中读出一些基本数据类型
* 作          者：lee
************************************************************/
public class ByteInputStream {
	
	//=====================================================
	//===================== Field =========================
	//=====================================================
	//缓冲器
	private byte[] mBuffer;
	//光标 
	private int mIndex = 0;

	/******************************************************
	* 功能描述：构造方法
	* @param buffer 要处理的数据
	*******************************************************/
	public ByteInputStream(byte[] buffer) {
		this.mBuffer = buffer;
	}
	
	/******************************************************
	* 功能描述：重新设置光标
	*******************************************************/
	public void resetCursor(){
		this.mIndex = 0;
	}
	
	/******************************************************
	* 功能描述：byte[4] 转换为 INT 读一个INT型 光标后移4字节
	*******************************************************/
	public int readInt(){
		try{
			int num = 0;
			for (int i = mIndex; i < (mIndex + 4); i++) {
				num <<= 8;
				num |= (mBuffer[i] & 0xff);
			}
			//光标移动4位
			mIndex += 4;
			return num;
		}catch (Exception e) {
			LOG.E("readInt" , e);
			return 0;
		}
	}
	
	/******************************************************
	* 功能描述：byte[2] 转换为 SHORT 读一个SHORT型 光标后移2字节
	*******************************************************/
	public short readShort() {
		try{
			short v = (short)(mBuffer[(mIndex + 1)] & 0xff | (mBuffer[mIndex] & 0xff) << 8);
			mIndex += 2;
			return v;
		}catch (Exception e) {
			LOG.E("readShort" , e);
			return 0;
		}
	}
	
	/******************************************************
	* 功能描述：读一个字节 光标后移1字节
	*******************************************************/
	public byte readByte(){
		try{
			return mBuffer[mIndex++];
		}catch (Exception e) {
			LOG.E("readByte" , e);
			return 0;
		}
	}
	
	/******************************************************
	* 功能描述：读字符串，首先读一个INT为长度，再读字符串
	*******************************************************/
	public String readUTF()
	{
		try{
			//读长度
			int size = 0;
			for (int i = mIndex; i < (mIndex + 4); i++) {
				size <<= 8;
				size |= (mBuffer[i] & 0xff);
			}
			mIndex += 4;
			
			//读字符串
			String value = new String(mBuffer, mIndex, size, "UTF8");
			mIndex += size;
			return value;
		}
		catch (Exception e) 
		{
			LOG.E("readUTF" , e);
			return null;
		}
	}
}
