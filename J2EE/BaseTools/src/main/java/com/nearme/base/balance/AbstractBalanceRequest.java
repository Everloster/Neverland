/**
 * AbstractBalanceRequest.java
 * com.nearme.base.balance
 *
 * Function： TODO
 *
 *   ver     date      		author
 * ──────────────────────────────────
 *   1.0	 2013-3-19 		80036381
 *
 * Copyright (c) 2013 NearMe, All Rights Reserved.
*/

package com.nearme.base.balance;

import java.io.File;
import java.io.InputStream;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;

import com.nearme.base.balance.event.ServerRemoveListener;
import com.nearme.base.balance.model.ServerConfig;
import com.nearme.base.balance.model.ServerGroupInfo;
import com.nearme.base.balance.model.ServerInfo;
import com.nearme.base.balance.parser.ServerGroupParser;
import com.nearme.base.balance.strategy.IConfigLoadStrategy;
import com.nearme.base.balance.strategy.InputStreamConfigLoadParser;
import com.nearme.base.balance.strategy.XmlFileConfigLoadParser;

/**
 * ClassName:AbstractBalanceRequest <br>
 * Function: 均衡的请求 <br>
 *
 * @author   80036381
 * @version
 * @since    Ver 1.1
 * @Date	 2013-3-19  下午5:42:33
 */
public abstract class AbstractBalanceRequest {
	private ServerConfig serverConfig;	//
	private IConfigLoadStrategy configLoadStrategy;	//配置加载策略
	private Random rand = new Random();

	/**
	 * 文件方式的加载，此处默认为xml文件
	 *
	 * @param configFile
	 */
	protected void init(File configFile) {
		ServerGroupParser configParser = new ServerGroupParser();
		this.configLoadStrategy = new XmlFileConfigLoadParser(configParser, configFile);
	}

	/**
	 * InputStream方式的加载，此处默认为xml的InputStream
	 *
	 * @param inputStream
	 */
	protected void init(InputStream inputStream) {
		ServerGroupParser configParser = new ServerGroupParser();
		this.configLoadStrategy = new InputStreamConfigLoadParser(configParser, inputStream);
	}

	/**
	 * 添加服务移除监听器
	 * @param
	 * @return
	 */
	public void addServerRemoveListener(ServerRemoveListener serverRemoveListener) {
		ExecuteStatistics.getInstance().addServerRemoveListener(serverRemoveListener);
	}

	/**
	 * 移除服务移除监听器
	 * @param
	 * @return
	 */
	public void removeServerRemoveListener(ServerRemoveListener serverRemoveListener) {
		ExecuteStatistics.getInstance().removeServerRemoveListener(serverRemoveListener);
	}

	//执行统计
	protected void stat(String groupId, ServerInfo server, boolean succ) {
		ExecuteStatistics.getInstance().addExecuteResult(groupId, server, succ);
	}

	/**
	 * 根据组id获取对应的服务信息
	 * @param
	 * @return
	 * @throws Exception
	 */
	protected ServerInfo getConnectServer(String groupId) throws Exception {
		List<ServerInfo> servers = getAllServers(groupId);

		int size = null == servers ? 0 : servers.size();

		if(size == 0) {
			return null;
		} else if(size == 1) {
			return servers.get(0);
		} else {
			//按照权重获取server
			int totalWeight = servers.get(0).getServerGroup().getTotalWeight();
			int weight = rand.nextInt(totalWeight);
			int tmpWeight = 0;
			for(int i = 0; i < size; i++) {
				ServerInfo server = servers.get(i);
				tmpWeight += server.getWeight();
				if(weight < tmpWeight) {
					return server;
				}
			}

			//随机获取server
			return servers.get(rand.nextInt(size - 1));
		}
	}

	protected List<ServerInfo> getAllServers(String groupId) throws Exception {
		// 根据情况进行加载
		boolean reload = false;
		if (needLoad()) {
			serverConfig = configLoadStrategy.parse();
			reload = true;
		}

		if (null == serverConfig) {
			return null;
		}

		ServerGroupInfo group = serverConfig.getServerGroup(groupId);
		if (null == group) {
			return null;
		}

		// 如果存在服务则直接返回
		List<ServerInfo> existsServers = group.getServerList();
		if (null == existsServers) {
			return null;
		}

		// 防止中途被移除，先拷贝出来
		List<ServerInfo> servers = new ArrayList<ServerInfo>(existsServers);
		// 如果不是新加载的且可自动启用服务则尝试重新加载
		if (!reload && servers.size() == 0 && group.isAutoRestart()) {
			ServerGroupInfo sgi = configLoadStrategy.parseWithId(group.getId());
			if (null != sgi) {
				serverConfig.addServerGroup(sgi);
				existsServers = sgi.getServerList();
				if (null != existsServers) {
					servers = new ArrayList<ServerInfo>(existsServers);
				}
			}
		}

		return servers;
	}

	/**
	 * 是否需要加载配置文件
	 * @param
	 * @return
	 */
	protected boolean needLoad() {
		if(null == serverConfig) {
			return true;
		}

		return configLoadStrategy.needLoad();
	}
}

