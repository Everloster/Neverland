/**
 * NormalMailSender.java
 * com.oppo.base.mail
 *
 * Function： TODO
 *
 *   ver     date      		author
 * ──────────────────────────────────
 *   1.0	 2011-5-27 		80036381
 *
 * Copyright (c) 2011 OPPO, All Rights Reserved.
*/

package com.oppo.base.mail;

import java.io.IOException;
import java.util.Date;
import java.util.HashMap;
import java.util.Map;
import java.util.Properties;

import javax.mail.Authenticator;
import javax.mail.MessagingException;
import javax.mail.Multipart;
import javax.mail.PasswordAuthentication;
import javax.mail.Session;
import javax.mail.Transport;
import javax.mail.internet.AddressException;
import javax.mail.internet.InternetAddress;
import javax.mail.internet.MimeBodyPart;
import javax.mail.internet.MimeMessage;
import javax.mail.internet.MimeMultipart;
import javax.mail.internet.MimeUtility;

import com.oppo.base.common.StringUtil;

/**
 * ClassName:SmtpMailSender
 * Function: SMTP邮件发送
 * 设置中需要有以下几项：
 * mail.smtp.host
 * mail.smtp.port
 * mail.smtp.auth
 * mail.smtp.user
 * mail.smtp.password
 * mail.smtp.from
 *
 * 使用方式：
 *
 * //初始化配置
 * Map<String, String> configMap = new HashMap<String, String>();
 * configMap.put("mail.smtp.host", "smtp.163.com");
 * configMap.put("mail.smtp.auth", "true");
 * configMap.put("mail.smtp.port", "25");
 * configMap.put("mail.smtp.user", "youaremoon");
 * configMap.put("mail.smtp.from", "youaremoon@163.com");
 * configMap.put("mail.smtp.password", "xxxxxxx");
 *
 * //设置初始化配置
 * SmtpMailSender normalMailSender = new SmtpMailSender();
 * normalMailSender.initial(configMap);
 * //方式二：直接初始化
 * //normalMailSender.initial("smtp.163.com", 25, true, "youaremoon", "xxxxxxx", "youaremoon@163.com");
 *
 * //发送邮件并返回相关代码
 * int code = normalMailSender.sendMultipartMail("", "53469290@qq.com", "发送测试1", "中文正文1",
 * 			new String[]{"G:\\build.xml","g:\\测试.htm"});
 * System.out.println(code);
 *
 * @author   80036381
 * @version
 * @since    Ver 1.1
 * @Date	 2011-5-27  下午05:14:54
 */
public class SmtpMailSender {
	/**
	 * 邮件发送成功
	 */
	public static final int SUCCESS = 1;

	/**
	 * 邮件地址错误
	 */
	public static final int ADDRESS_EXCEPTION = -1;

	/**
	 * 邮件发送错误
	 */
	public static final int MESSAGING_EXCEPTION = -2;

	/**
	 * 附件读取错误
	 */
	public static final int MULTIPART_EXCEPTION = -3;

	/**
	 * 附件空
	 */
	public static final int MULTIPART_EMPTY = -4;

	public static final int SEND_HTML = 0;
	public static final int SEND_TEXT = 1;

	private int sendType;
	private Session session;

	public SmtpMailSender() {

	}

	/**
	 * 使用map初始化基本配置
	 * @param
	 * @return
	 */
	public void initial(Map<String, String> configMap) {
		Properties prop = new Properties();
		for(String key : configMap.keySet()) {
			prop.setProperty(key, configMap.get(key));
		}

		initial(prop);
	}

	/**
	 * 初始化基本配置
	 * @param host smtp域名
	 * @param port 端口
	 * @param auth 是否进行用户验证
	 * @param userName 用户名
	 * @param password 密码
	 * @param from 用户名对应邮箱
	 * @return
	 */
	public void initial(String host, int port, boolean auth, String userName, String password, String from) {
		Properties prop = new Properties();
		prop.put("mail.smtp.host", host);
		prop.put("mail.smtp.port", port);
		prop.put("mail.smtp.auth", auth);
		prop.put("mail.smtp.user", userName);
		prop.put("mail.smtp.password", password);
		prop.put("mail.smtp.from", from);

		initial(prop);
	}

	/**
	 * 初始化基本配置
	 * @param
	 * @return
	 */
	public void initial(final Properties prop) {
		prop.setProperty("mail.transport.protocol", "smtp");
		session = Session.getInstance(prop, new Authenticator() {
			public PasswordAuthentication getPasswordAuthentication() {
				return new PasswordAuthentication(
						prop.getProperty("mail.smtp.user"),
						prop.getProperty("mail.smtp.password"));
			}
		});
	}

	/**
	 * 发送普通邮件
	 * @param from
	 * @param to 接受邮件列表
	 * @param subject 主题
	 * @param content 邮件内容
	 * @return
	 */
	public int sendMail(String from, String to, String subject, String content) {
		return sendMultipartMail(from, to, subject, content, null);
	}

	/**
	 * 发送带附件的邮件
	 * Function Description here
	 * @param
	 * @return
	 */
	public int sendMultipartMail(String from, String to, String subject, String content, String[] filePaths) {
		Transport transport = null;
		MimeMessage mimeMessage = new MimeMessage(session);

		try {
			if(StringUtil.isNullOrEmpty(from)) {
				from = session.getProperty("mail.smtp.from");
			}

			//设置邮件基本信息
			mimeMessage.setFrom(new InternetAddress(from));
			mimeMessage.setSubject(MimeUtility.encodeText(subject));
			mimeMessage.addRecipients(MimeMessage.RecipientType.TO, InternetAddress.parse(to));

			//带附件邮件
			if(null != filePaths && filePaths.length > 0) {
				Multipart multipart = new MimeMultipart();

				//邮件附件
				for(int i = 0; i < filePaths.length; i++) {
					MimeBodyPart attachPart = new MimeBodyPart();
					attachPart.attachFile(filePaths[i]);
					multipart.addBodyPart(attachPart);
				}

				//邮件正文
				MimeBodyPart contentPart = new MimeBodyPart();
				if(sendType == SEND_HTML) {
					contentPart.setContent(content, "text/html;charset=utf-8");
				} else {
					contentPart.setText(content);
				}

				multipart.addBodyPart(contentPart);
				mimeMessage.setContent(multipart);
			} else {
				//正文
				if(sendType == SEND_HTML) {
					mimeMessage.setContent(content, "text/html;charset=utf-8");
				} else {
					mimeMessage.setText(content);
				}
			}

			//发送时间
			mimeMessage.setSentDate(new Date());

			//连接邮件服务器
			transport = session.getTransport();
        	transport.connect();

	        // 发送邮件,其中第二个参数是所有已设好的收件人地址
	        transport.sendMessage(mimeMessage, mimeMessage.getAllRecipients());

	        return SUCCESS;
		} catch (AddressException e) {
			e.printStackTrace();
			return ADDRESS_EXCEPTION;
		} catch (IOException e) {
			e.printStackTrace();
			return MULTIPART_EXCEPTION;
		} catch (MessagingException e) {
			e.printStackTrace();
			return MESSAGING_EXCEPTION;
		} finally {
			if(null != transport) {
				try {
					transport.close();
				} catch(Exception ex) {
				}
			}
		}
	}

	/**
	 * 获取sendType
	 * @return  the sendType
	 * @since   Ver 1.0
	 */
	public int getSendType() {
		return sendType;
	}

	/**
	 * 设置sendType
	 * @param   sendType
	 * @since   Ver 1.0
	 */
	public void setSendType(int sendType) {
		this.sendType = sendType;
	}

	public static void main(String[] args) throws Exception {
		Map<String, String> configMap = new HashMap<String, String>();
		configMap.put("mail.smtp.host", "smtp.126.com");
		configMap.put("mail.smtp.auth", "true");
		configMap.put("mail.smtp.port", "25");
		configMap.put("mail.smtp.user", "youaremoon");
		configMap.put("mail.smtp.from", "youaremoon@126.com");
		configMap.put("mail.smtp.password", "iloveyou850910");

		SmtpMailSender normalMailSender = new SmtpMailSender();
		normalMailSender.initial(configMap);
		//normalMailSender.initial("smtp.163.com", 25, true, "youaremoon", "iloveyou850726", "youaremoon@163.com");
		String send = "发送中文内容了";
		int code = normalMailSender.sendMultipartMail("", "youaremoon@126.com", "发送测试1", send,
			null);
		System.out.println(code);
	}
}

