/**
 * BusinessServerInfo.java
 * com.nearme.base.netty.connect
 *
 * Function： TODO 
 *
 *   ver     date      		author
 * ──────────────────────────────────
 *   1.0	 2012-12-7 		80036381
 *
 * Copyright (c) 2012 NearMe, All Rights Reserved.
*/

package com.nearme.base.netty.connect;

import java.io.Serializable;

/**
 * ClassName:BusinessServerInfo <br>
 * Function: 业务服务器信息 <br>
 *
 * @author   80036381
 * @version  
 * @since    Ver 1.1
 * @Date	 2012-12-7  上午10:47:04
 */
public class BusinessServerInfo implements Serializable {
	/**
	 * @since Ver 1.1
	 */
	private static final long serialVersionUID = 1L;
	
	private String serverId;	//业务服务器id
	private String groupId;		//业务服务器所属组id
	private String serverIp;	//业务服务器ip
	private int serverPort;		//业务服务器端口
	private double weight;		//业务服务器权重
	/**
	 * 获取业务服务器id
	 * @return  the serverId
	 * @since   Ver 1.0
	 */
	public String getServerId() {
		return serverId;
	}
	/**
	 * 设置业务服务器id
	 * @param   serverId    
	 * @since   Ver 1.0
	 */
	public void setServerId(String serverId) {
		this.serverId = serverId;
	}
	/**
	 * 获取业务服务器所属组id
	 * @return  the groupId
	 * @since   Ver 1.0
	 */
	public String getGroupId() {
		return groupId;
	}
	/**
	 * 设置业务服务器所属组id
	 * @param   groupId    
	 * @since   Ver 1.0
	 */
	public void setGroupId(String groupId) {
		this.groupId = groupId;
	}
	/**
	 * 获取业务服务器ip
	 * @return  the serverIp
	 * @since   Ver 1.0
	 */
	public String getServerIp() {
		return serverIp;
	}
	/**
	 * 设置业务服务器ip
	 * @param   serverIp    
	 * @since   Ver 1.0
	 */
	public void setServerIp(String serverIp) {
		this.serverIp = serverIp;
	}
	/**
	 * 获取业务服务器端口
	 * @return  the serverPort
	 * @since   Ver 1.0
	 */
	public int getServerPort() {
		return serverPort;
	}
	/**
	 * 设置业务服务器端口
	 * @param   serverPort    
	 * @since   Ver 1.0
	 */
	public void setServerPort(int serverPort) {
		this.serverPort = serverPort;
	}
	/**
	 * 获取业务服务器权重
	 * @return  the weight
	 * @since   Ver 1.0
	 */
	public double getWeight() {
		return weight;
	}
	/**
	 * 设置业务服务器权重
	 * @param   weight    
	 * @since   Ver 1.0
	 */
	public void setWeight(double weight) {
		this.weight = weight;
	}
}

