/**
 * ExecuteInfo.java
 * com.nearme.base.balance.model
 *
 * Function： TODO
 *
 *   ver     date      		author
 * ──────────────────────────────────
 *   1.0	 2012-8-15 		80036381
 *
 * Copyright (c) 2012 NearMe, All Rights Reserved.
*/

package com.nearme.base.balance.model;

import java.io.Serializable;

/**
 * ClassName:ExecuteInfo <br>
 * Function: 执行信息 <br>
 *
 * @author   80036381
 * @version
 * @since    Ver 1.1
 * @Date	 2012-8-15  下午08:48:48
 */
public class ExecuteInfo implements Serializable {
	/**
	 * @since Ver 1.1
	 */
	private static final long serialVersionUID = 1L;
	private ServerInfo serverInfo;
	private int success;
	private int fail;
	private int total;

	public ExecuteInfo(ServerInfo serverInfo) {
		this.serverInfo = serverInfo;
		reset();
	}

	/**
	 * 添加一次成功的次数
	 * @param
	 * @return
	 */
	public void addSuccess() {
		success++;
		total++;
	}

	/**
	 * 添加一次失败的次数
	 * @param
	 * @return
	 */
	public void addFail() {
		fail++;
		total++;
	}

	/**
	 * 重置成功失败次数
	 * @param
	 * @return
	 */
	public void reset() {
		this.success = 0;
		this.fail = 0;
		this.total = 0;
	}

	/**
	 *
	 * @param
	 * @return
	 */
	public ServerInfo getServerInfo() {
		return serverInfo;
	}

	/**
	 * 获取success
	 * @return  the success
	 * @since   Ver 1.0
	 */
	public int getSuccess() {
		return success;
	}

	/**
	 * 获取fail
	 * @return  the fail
	 * @since   Ver 1.0
	 */
	public int getFail() {
		return fail;
	}

	/**
	 * 获取总执行次数
	 * @param
	 * @return
	 */
	public int getTotal() {
		return total;
	}
}

