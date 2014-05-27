/************************************************************
* 文  件  名：ByteOutputStream.java
* 功能模块：控制
* 功能描述：把基本数据类型变成一个Byte数组，可以组装int short byte 字符串
* 作          者：lee
************************************************************/
package nio.game.server;

import nio.game.utils.LOG;
import java.io.ByteArrayOutputStream;
import java.io.IOException;

/************************************************************
* 类          名：ByteOutputStream
* 功能描述：把基本数据类型变成一个Byte数组，可以组装int short byte 字符串
* 作          者：lee
************************************************************/
public class ByteOutputStream {
	
	//=====================================================
	//===================== Field =========================
	//=====================================================
	private ByteArrayOutputStream mOut; //缓冲器
	
	/******************************************************
	* 功能描述：构造方法 ，初始化缓冲器
	*******************************************************/
	public ByteOutputStream()
	{
		this.mOut = new ByteArrayOutputStream(512);
	}
	
	/******************************************************
	* 功能描述：完成写入，返回最终的数据流，一旦调用flush将不可再用
	* @return 数据
	*******************************************************/
	public byte[] flush()
	{
		byte[] buff = this.mOut.toByteArray();
		try {
			this.mOut.close();
		} catch (IOException e) {
			LOG.E("flush()",e);
		}
		return buff;
	}
	
	/******************************************************
	* 功能描述：写一个4字节INT
	* @param 值
	*******************************************************/
	public void writeInt(int num){
		//4字节INT转为byte[4];
		byte[] b = new byte[4];
		for (int i = 0; i < 4; i++) {
			b[i] = (byte) (num >>> (24 - i * 8));
		}
		//写4字节
		try {
			this.mOut.write(b);
		} catch (IOException e) {
			LOG.E("writeInt", e);
		}
	}
	
	/******************************************************
	* 功能描述：写一个2字节SHORT
	* @param 值
	*******************************************************/
	public void writeShort(short num){
		byte[] b = new byte[2];
		b[1] = (byte) (num & 0xff);
		b[0] = (byte) ((num >> 8) & 0xff);
		
		//写2字节
		try {
			this.mOut.write(b);
		} catch (IOException e) {
			LOG.E("writeShort", e);
		}
	}
	
	/******************************************************
	* 功能描述：写单字节
	* @param 值
	*******************************************************/
	public void writeByte(byte mun){
		this.mOut.write((int)mun);
	}
	
	/******************************************************
	* 功能描述：写一个字符串，首先写长度INT4字节
	* @param 值
	*******************************************************/
	public void writeUTF(String str){
		if(str != null && str.length() > 0){
			try{
				byte[] buff = str.getBytes("UTF8");
				
				int size = buff.length;
				//4字节int转为byte[4];
				byte[] b = new byte[4];
				for (int i = 0; i < 4; i++) {
					b[i] = (byte) (size >>> (24 - i * 8));
				}
				
				//写入4字节长度
				this.mOut.write(b);
				//写入字符串
				this.mOut.write(buff);
			}catch (Exception e) {
				e.printStackTrace();
			}
		}
		else{
			LOG.E("writeUTF if(str != null && str.length() > 0){");
		}
	}
	
	/******************************************************
	* 功能描述：把一个4字节INT型直接转为byte数组
	* @param 值
	*******************************************************/
	public static byte[] writeIntToBytes(int num){
		//4字节INT转为byte[4];
		byte[] b = new byte[4];
		for (int i = 0; i < 4; i++) {
			b[i] = (byte) (num >>> (24 - i * 8));
		}
		return b;
	}

}
