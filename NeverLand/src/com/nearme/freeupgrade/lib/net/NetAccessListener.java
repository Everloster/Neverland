package com.nearme.freeupgrade.lib.net;

/**
 * @author 80054358
 * 
 *         网络接口访问回调接口
 */
public interface NetAccessListener {

	public void onNetAccessFail(int opCode, String errorMsg);

	public void onNetAccessSuccess(int opCode, Object data);

}
