/**
 * ServerGroupParser.java
 * com.nearme.base.balance.parser
 *
 * Function： TODO
 *
 *   ver     date      		author
 * ──────────────────────────────────
 *   1.0	 2012-8-15 		80036381
 *
 * Copyright (c) 2012 NearMe, All Rights Reserved.
*/

package com.nearme.base.balance.parser;

import java.io.ByteArrayInputStream;
import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.io.UnsupportedEncodingException;
import java.util.ArrayList;
import java.util.List;

import javax.xml.parsers.DocumentBuilder;
import javax.xml.parsers.DocumentBuilderFactory;
import javax.xml.parsers.ParserConfigurationException;

import org.w3c.dom.Document;
import org.w3c.dom.Element;
import org.w3c.dom.NamedNodeMap;
import org.w3c.dom.Node;
import org.w3c.dom.NodeList;
import org.xml.sax.SAXException;

import com.nearme.base.balance.model.ServerConfig;
import com.nearme.base.balance.model.ServerGroupInfo;
import com.nearme.base.balance.model.ServerInfo;
import com.oppo.base.common.NumericUtil;
import com.oppo.base.common.OConstants;

/**
 * ClassName:ServerGroupParser <br>
 * Function: 服务文件配置解析 <br>
 *
 * @author   80036381
 * @version
 * @since    Ver 1.1
 * @Date	 2012-8-15  下午05:08:21
 */
public class ServerGroupParser implements IConfigParser {

	private static final String NODE_SERVER_GROUP = "server-group";
	private static final String NODE_SERVER = "server";

	/**
	 * 解析指定的xml对象得到server配置
	 * @param xmlObject 支持File,xml String,InputStream类型
	 * @return
	 */
	@Override
	public ServerConfig parse(Object xmlObject) throws Exception {
		return visitRoot(getDocumentElement(xmlObject));
	}

	/**
	 * 解析指定的xml对象得到指定id的服务组信息
	 * @param xmlObject 支持File,xml String,InputStream类型
	 * @return
	 */
	@Override
	public ServerGroupInfo parseWithId(Object xmlObject, String id) throws Exception {
		if(null == id) {
			throw new NullPointerException("id cann't be null");
		}

		Element root = getDocumentElement(xmlObject);
		NodeList nodeList = root.getChildNodes();

		for(int i = 0, len = nodeList.getLength(); i < len; i++) {
			Node node = nodeList.item(i);
			String nodeName = node.getNodeName();
			if(NODE_SERVER_GROUP.equals(nodeName)) {
				ServerGroupInfo group = getServerGroup(node, id);
				if(null != group) {
					return group;
				}
			}
		}

		return null;
	}

	private Element getDocumentElement(Object xmlObject)
		throws ParserConfigurationException, UnsupportedEncodingException, SAXException, IOException {
		DocumentBuilderFactory dbf = DocumentBuilderFactory.newInstance();

		DocumentBuilder db = dbf.newDocumentBuilder();
		Document doc = null;
		if(xmlObject instanceof String){
			doc = db.parse(new ByteArrayInputStream(xmlObject.toString().getBytes(OConstants.DEFAULT_ENCODING)));
		} else if(xmlObject instanceof File){
			doc = db.parse((File)xmlObject);
		} else if(xmlObject instanceof InputStream) {
			doc = db.parse((InputStream)xmlObject);
		} else {
			throw new IllegalArgumentException();
		}

		return doc.getDocumentElement();
	}

	//遍历根节点,解析得到配置文件
	private ServerConfig visitRoot(Element element) {
		NodeList nodeList = element.getChildNodes();

		ServerConfig sc = new ServerConfig();
		for(int i = 0, len = nodeList.getLength(); i < len; i++) {
			Node node = nodeList.item(i);
			String nodeName = node.getNodeName();
			if(NODE_SERVER_GROUP.equals(nodeName)) {
				ServerGroupInfo group = getServerGroup(node, null);
				if(null != group) {
					sc.addServerGroup(group);
				}
			}
		}

		return sc;
	}

	/**
	 * 解析获取服务组信息,当searchId不为空时，只有解析到id与此相同时才返回信息
	 */
	private ServerGroupInfo getServerGroup(Node node, String searchId) {
		//group基本属性
		NamedNodeMap map = node.getAttributes();
		String id = map.getNamedItem("id").getNodeValue();
		//当searchId不为空时，只有解析到的id与之匹配才返回信息
		if(null != searchId && !searchId.equals(id)) {
			return null;
		}

		ServerGroupInfo groupInfo = new ServerGroupInfo();
		groupInfo.setId(id);

		Node descNode = map.getNamedItem("desc");
		if(null != descNode) {
			groupInfo.setDescription(descNode.getNodeValue());
		}

		//当执行次数小于此值时不抛弃
		Node abandonMinSizeNode = map.getNamedItem("abandon-min-size");
		if(null != abandonMinSizeNode) {
			groupInfo.setAbandonMinSize(NumericUtil.parseInt(abandonMinSizeNode.getNodeValue(),
					ServerGroupInfo.DEFAULT_ABANDON_MIN_SIZE));
		}

		//当失败率达到此比例时，服务将被抛弃（结合abandon-min-size）
		Node abandonRateNode = map.getNamedItem("abandon-rate");
		if(null != abandonRateNode) {
			groupInfo.setAbandonRate(NumericUtil.parseDouble(abandonRateNode.getNodeValue(),
					ServerGroupInfo.DEFAULT_ABANDON_RATE));
		}

		//是否自动重新开启服务
		Node autoRestartNode = map.getNamedItem("auto-restart");
		if(null != autoRestartNode) {
			groupInfo.setAutoRestart(Boolean.parseBoolean(autoRestartNode.getNodeValue()));
		}

		//server信息
		NodeList childNodes = node.getChildNodes();
		int len = childNodes.getLength();
		if(len >= 1) {
			List<ServerInfo> servers = new ArrayList<ServerInfo>(len);
			for(int i = 0; i < len; i++) {
				Node childNode = childNodes.item(i);
				String nodeName = childNode.getNodeName();
				//解析server节点
				if(NODE_SERVER.equals(nodeName)) {
					ServerInfo serverInfo = getServerInfo(childNode);
					//只加入激活的服务
					if(null != serverInfo && serverInfo.isActive()) {
						servers.add(serverInfo);
					}
				}
			}

			groupInfo.setServerList(servers);
		}

		return groupInfo;
	}

	//解析获取单个服务信息
	private ServerInfo getServerInfo(Node node) {
		ServerInfo serverInfo = new ServerInfo();

		//server基本属性
		NamedNodeMap map = node.getAttributes();

		//id节点必须有
		serverInfo.setServerId(map.getNamedItem("id").getNodeValue());

		Node descNode = map.getNamedItem("desc");
		if(null != descNode) {
			serverInfo.setDescription(descNode.getNodeValue());
		}

		//server-url节点必须有
		serverInfo.setServerUrl(map.getNamedItem("server-url").getNodeValue());

		//weight节点必须有
		int weight = NumericUtil.parseInt(map.getNamedItem("weight").getNodeValue(), 1);
		if(weight <= 0) {
			weight = 1;
		}
		serverInfo.setWeight(weight);

		Node hWeightNode = map.getNamedItem("highest-weight");
		int highestWeight = 0;
		if(null != hWeightNode) {
			highestWeight = NumericUtil.parseInt(hWeightNode.getNodeValue(), 0);
		}
		if(highestWeight < weight) {
			highestWeight = weight;
		}
		serverInfo.setHighestWeight(highestWeight);

		//是否是新加入的节点，默认不是
		Node isNewNode = map.getNamedItem("is-new");
		if(null != isNewNode) {
			serverInfo.setNew(Boolean.parseBoolean(isNewNode.getNodeValue()));
		}

		//是否激活，默认不激活
		Node isActiveNode = map.getNamedItem("active");
		if(null != isActiveNode) {
			serverInfo.setActive(Boolean.parseBoolean(isActiveNode.getNodeValue()));
		}

		return serverInfo;
	}

	public static void main(String[] args) throws Exception {
		ServerGroupParser sgp = new ServerGroupParser();
		ServerConfig sc = sgp.parse("<c><server-group id='group1' desc='连接推荐服务器'><server id='server1' desc='hz001' server-url='http://a.com:80/' weight='80' is-new='false' active='true' /></server-group></c>");
		System.out.println(sc.getServerGroups().get(0).getDescription());
	}
}

