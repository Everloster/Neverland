/************************************************************
* 文  件  名：LOG.java
* 功能模块：日志
* 功能描述：定义了所有日志信息的打印格式
* 作          者：lee
************************************************************/
package nio.game.utils;

/************************************************************
* 类          名：LOG
* 功能描述：定义了所有日志信息的打印格式
* 作          者：lee
************************************************************/
public final class LOG {
	
	//调试信息的头部信息
	public static final String LOG_TITLE = "--# [game] : ";

	/*******************************************************
	* 功能描述：DEBUG调试信息输出
	* @param msg 调试信息
	*******************************************************/
	public static void D(String msg){
		System.out.println(LOG_TITLE + msg);
	}

	/*******************************************************
	* 功能描述：错误调试信息输出
	* @param msg 错误消息
	*******************************************************/
	public static void E(String msg){
		System.err.println(LOG_TITLE + msg);
	}

	/*******************************************************
	* 功能描述：错误调试信息输出，并附带错误信息
	* @param msg 错误消息
	* @param e 异常实例
	*******************************************************/
	public static void E(String msg , Exception e){
		System.err.println(new StringBuffer()
		.append(LOG_TITLE)
		.append(msg)
		.append(" - ")
		.append(e.toString())
		.toString());
	}

}
