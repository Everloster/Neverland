package org.jabe.neverland.pattern;


/**
 * 命令模式, 不太明白.
 * 
 * 将一个请求封装为一个对象，从而使你可用不同的请求对客户进行参数化；对请求排队或记录请求日志，以及支持可撤消的操作。
 * 
 *   1.抽象出待执行的动作以参数化某对象。
 *   2.在不同的时刻指定、排列和执行请求。
 *   3.支持取消操作。
 *   4.支持修改日志，这样当系统崩溃时，这些修改可以被重做一遍。
 *   5.用构建在原语操作上的高层操作构造一个系统。
 *   
 * 
 * @author Jabe
 */
public class CommandPattern {
	/**
	 *  声明执行操作的接口。
	 *  
	 * @author Jabe
	 */
	public static abstract class Command {
		protected Receiver receiver;

		public Command(Receiver receiver) {
			this.receiver = receiver;
		}

		public abstract void execute();

	}

	/**
	 * 将一个接收者对象绑定于一个动作。  调用接收者相应的操作，以实现Execute。
	 * 
	 * @author Jabe
	 */
	public static class CommandImpl extends Command {
		public CommandImpl(Receiver receiver) {
			super(receiver);
		}

		public void execute() {
			receiver.receive();
		}

	}

	/**
	 * 要求该命令执行这个请求。
	 * 
	 * @author Jabe
	 */
	public static class Invoker {

		private Command command;

		public void setCommand(Command command) {
			this.command = command;
		}

		public void execute() {
			command.execute();
		}
	}

	/**
	 * 知道如何实施与执行一个请求相关的操作。任何类都可能作为一个接收者。
	 * 
	 * @author Jabe
	 */
	public static class Receiver {

		public void receive() {
			System.out.println("This is Receive class!");
		}
	}

	public static void main(String[] args) {
		Receiver rec = new Receiver();
		Command cmd = new CommandImpl(rec);
		Invoker i = new Invoker();
		i.setCommand(cmd);
		i.execute();
	}

}
