/**
 * ServerInfo.java
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
 * ClassName:ServerInfo <br>
 * Function: 服务信息 <br>
 *
 * @author   80036381
 * @version
 * @since    Ver 1.1
 * @Date	 2012-8-15  下午05:00:39
 */
public class ServerInfo implements Serializable {
	/**
	 * @since Ver 1.1
	 */
	private static final long serialVersionUID = 1L;

	private String serverId;	//服务id
	private String description;	//描述
	private String serverUrl;	//server对应的url
	private int highestWeight; 	//最高权重
	private int weight;			//权重
	private boolean isNew;		//是否是新上线
	private boolean isActive;	//是否是激活的

	private ServerGroupInfo serverGroup; //服务所在组

	/**
	 * 获取服务id
	 * @return  the serverId
	 * @since   Ver 1.0
	 */
	public String getServerId() {
		return serverId;
	}
	/**
	 * 设置服务id
	 * @param   serverId
	 * @since   Ver 1.0
	 */
	public void setServerId(String serverId) {
		this.serverId = serverId;
	}
	/**
	 * 获取描述
	 * @return  the description
	 * @since   Ver 1.0
	 */
	public String getDescription() {
		return description;
	}
	/**
	 * 设置描述
	 * @param   description
	 * @since   Ver 1.0
	 */
	public void setDescription(String description) {
		this.description = description;
	}
	/**
	 * 获取server对应的url
	 * @return  the serverUrl
	 * @since   Ver 1.0
	 */
	public String getServerUrl() {
		return serverUrl;
	}
	/**
	 * 设置server对应的url
	 * @param   serverUrl
	 * @since   Ver 1.0
	 */
	public void setServerUrl(String serverUrl) {
		this.serverUrl = serverUrl;
	}
	/**
	 * 获取权重
	 * @return  the weight
	 * @since   Ver 1.0
	 */
	public int getWeight() {
		return weight;
	}
	/**
	 * 设置权重
	 * @param   weight
	 * @since   Ver 1.0
	 */
	public void setWeight(int weight) {
		this.weight = weight;
	}
	/**
	 * 获取最高
	 * @return  the highestWeight
	 * @since   Ver 1.0
	 */
	public int getHighestWeight() {
		return highestWeight;
	}
	/**
	 * 设置最高
	 * @param   highestWeight
	 * @since   Ver 1.0
	 */
	public void setHighestWeight(int highestWeight) {
		this.highestWeight = highestWeight;
	}
	/**
	 * 获取是否是新上线
	 * @return  the isNew
	 * @since   Ver 1.0
	 */
	public boolean isNew() {
		return isNew;
	}
	/**
	 * 设置是否是新上线
	 * @param   isNew
	 * @since   Ver 1.0
	 */
	public void setNew(boolean isNew) {
		this.isNew = isNew;
	}
	/**
	 * 获取是否是激活的
	 * @return  the isActive
	 * @since   Ver 1.0
	 */
	public boolean isActive() {
		return isActive;
	}
	/**
	 * 设置是否是激活的
	 * @param   isActive
	 * @since   Ver 1.0
	 */
	public void setActive(boolean isActive) {
		this.isActive = isActive;
	}
	/**
	 * 获取serverGroup
	 * @return  the serverGroup
	 * @since   Ver 1.0
	 */
	public ServerGroupInfo getServerGroup() {
		return serverGroup;
	}
	/**
	 * 设置serverGroup
	 * @param   serverGroup
	 * @since   Ver 1.0
	 */
	public void setServerGroup(ServerGroupInfo serverGroup) {
		this.serverGroup = serverGroup;
	}
}

