package org.jabe.neverland.px2dp;

import java.awt.EventQueue;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.io.File;
import java.util.logging.Level;
import java.util.logging.Logger;

import javax.swing.GroupLayout;
import javax.swing.JButton;
import javax.swing.JFileChooser;
import javax.swing.JFrame;
import javax.swing.JLabel;
import javax.swing.JOptionPane;
import javax.swing.JSeparator;
import javax.swing.JTextField;
import javax.swing.LayoutStyle;
import javax.swing.UIManager;
import javax.swing.UnsupportedLookAndFeelException;

public class NewJFrame extends JFrame {
	private JButton btn_1;
	private JButton btn_2;
	private JButton btn_3;
	private JButton btn_4;
	private JButton jButton3;
	private JButton jButton5;
	private JLabel jLabel1;
	private JLabel jLabel2;
	private JLabel jLabel3;
	private JLabel jLabel4;
	private JLabel jLabel5;
	private JLabel jLabel6;
	private JSeparator jSeparator1;
	private JTextField text_1;
	private JTextField text_2;
	private JTextField text_3;
	private JTextField text_4;

	public NewJFrame() {
		initComponents();
		setLocationRelativeTo(null);
	}

	private void initComponents() {
		this.jLabel1 = new JLabel();
		this.jLabel2 = new JLabel();
		this.text_1 = new JTextField();
		this.text_2 = new JTextField();
		this.btn_1 = new JButton();
		this.btn_2 = new JButton();
		this.jButton3 = new JButton();
		this.jLabel3 = new JLabel();
		this.jSeparator1 = new JSeparator();
		this.text_3 = new JTextField();
		this.jLabel4 = new JLabel();
		this.jLabel5 = new JLabel();
		this.jLabel6 = new JLabel();
		this.btn_4 = new JButton();
		this.jButton5 = new JButton();
		this.text_4 = new JTextField();
		this.btn_3 = new JButton();

		setDefaultCloseOperation(3);
		setTitle("Px2Dp");
		setName("Px2Dp");
		setResizable(false);

		this.jLabel1.setText("源文件路径");

		this.jLabel2.setText("保存路径");

		this.text_1.setEditable(false);

		this.text_2.setEditable(false);

		this.btn_1.setText("选择");
		this.btn_1.addActionListener(new ActionListener() {
			public void actionPerformed(ActionEvent evt) {
				NewJFrame.this.btn_1ActionPerformed(evt);
			}
		});
		this.btn_2.setText("选择");
		this.btn_2.addActionListener(new ActionListener() {
			public void actionPerformed(ActionEvent evt) {
				NewJFrame.this.btn_2ActionPerformed(evt);
			}
		});
		this.jButton3.setText("转换");
		this.jButton3.addActionListener(new ActionListener() {
			public void actionPerformed(ActionEvent evt) {
				NewJFrame.this.jButton3ActionPerformed(evt);
			}
		});
		this.jLabel3.setText("上面为Android布局文件DP批量转换为PX,分辨率为800*480");

		this.text_3.setEditable(false);

		this.jLabel4.setText("上面为Android样式文件PX批量转换为DP,px分辨率为800*480");

		this.jLabel5.setText("保存路径");

		this.jLabel6.setText("源文件路径");

		this.btn_4.setText("选择");
		this.btn_4.addActionListener(new ActionListener() {
			public void actionPerformed(ActionEvent evt) {
				NewJFrame.this.btn_4ActionPerformed(evt);
			}
		});
		this.jButton5.setText("转换");
		this.jButton5.addActionListener(new ActionListener() {
			public void actionPerformed(ActionEvent evt) {
				NewJFrame.this.jButton5ActionPerformed(evt);
			}
		});
		this.text_4.setEditable(false);

		this.btn_3.setText("选择");
		this.btn_3.addActionListener(new ActionListener() {
			public void actionPerformed(ActionEvent evt) {
				NewJFrame.this.btn_3ActionPerformed(evt);
			}
		});
		GroupLayout layout = new GroupLayout(getContentPane());
		getContentPane().setLayout(layout);
		layout.setHorizontalGroup(layout
				.createParallelGroup(GroupLayout.Alignment.LEADING)
				.addComponent(this.jSeparator1)
				.addGroup(
						layout.createSequentialGroup()
								.addContainerGap()
								.addGroup(
										layout.createParallelGroup(
												GroupLayout.Alignment.LEADING)
												.addGroup(
														layout.createSequentialGroup()
																.addGroup(
																		layout.createParallelGroup(
																				GroupLayout.Alignment.LEADING)
																				.addComponent(
																						this.jLabel1)
																				.addComponent(
																						this.jLabel2))
																.addGap(18, 18,
																		18)
																.addGroup(
																		layout.createParallelGroup(
																				GroupLayout.Alignment.LEADING,
																				false)
																				.addComponent(
																						this.text_1)
																				.addComponent(
																						this.text_2,
																						-1,
																						140,
																						32767))
																.addPreferredGap(
																		LayoutStyle.ComponentPlacement.RELATED)
																.addGroup(
																		layout.createParallelGroup(
																				GroupLayout.Alignment.LEADING)
																				.addComponent(
																						this.btn_2,
																						-1,
																						-1,
																						32767)
																				.addComponent(
																						this.btn_1,
																						-1,
																						-1,
																						32767))
																.addGap(18, 18,
																		18)
																.addComponent(
																		this.jButton3,
																		-2, 61,
																		-2))
												.addComponent(
														this.jLabel3,
														GroupLayout.Alignment.TRAILING,
														-1, -1, 32767)
												.addComponent(
														this.jLabel4,
														GroupLayout.Alignment.TRAILING,
														-1, -1, 32767)
												.addGroup(
														layout.createSequentialGroup()
																.addGroup(
																		layout.createParallelGroup(
																				GroupLayout.Alignment.LEADING)
																				.addComponent(
																						this.jLabel6)
																				.addComponent(
																						this.jLabel5))
																.addGap(18, 18,
																		18)
																.addGroup(
																		layout.createParallelGroup(
																				GroupLayout.Alignment.LEADING,
																				false)
																				.addComponent(
																						this.text_3)
																				.addComponent(
																						this.text_4,
																						-2,
																						140,
																						-2))
																.addPreferredGap(
																		LayoutStyle.ComponentPlacement.RELATED)
																.addGroup(
																		layout.createParallelGroup(
																				GroupLayout.Alignment.LEADING)
																				.addComponent(
																						this.btn_4,
																						-1,
																						77,
																						32767)
																				.addComponent(
																						this.btn_3,
																						-1,
																						-1,
																						32767))
																.addGap(18, 18,
																		18)
																.addComponent(
																		this.jButton5,
																		-2, 61,
																		-2)))
								.addContainerGap()));

		layout.setVerticalGroup(layout
				.createParallelGroup(GroupLayout.Alignment.LEADING)
				.addGroup(
						layout.createSequentialGroup()
								.addContainerGap()
								.addGroup(
										layout.createParallelGroup(
												GroupLayout.Alignment.LEADING)
												.addGroup(
														layout.createSequentialGroup()
																.addGroup(
																		layout.createParallelGroup(
																				GroupLayout.Alignment.BASELINE)
																				.addComponent(
																						this.jLabel1)
																				.addComponent(
																						this.text_1,
																						-2,
																						-1,
																						-2)
																				.addComponent(
																						this.btn_1))
																.addGap(18, 18,
																		18)
																.addGroup(
																		layout.createParallelGroup(
																				GroupLayout.Alignment.BASELINE)
																				.addComponent(
																						this.jLabel2)
																				.addComponent(
																						this.text_2,
																						-2,
																						-1,
																						-2)
																				.addComponent(
																						this.btn_2)))
												.addComponent(
														this.jButton3,
														GroupLayout.Alignment.TRAILING,
														-2, 50, -2))
								.addPreferredGap(
										LayoutStyle.ComponentPlacement.UNRELATED)
								.addComponent(this.jLabel3)
								.addPreferredGap(
										LayoutStyle.ComponentPlacement.RELATED)
								.addComponent(this.jSeparator1, -2, 10, -2)
								.addGroup(
										layout.createParallelGroup(
												GroupLayout.Alignment.LEADING)
												.addGroup(
														layout.createSequentialGroup()
																.addGroup(
																		layout.createParallelGroup(
																				GroupLayout.Alignment.BASELINE)
																				.addComponent(
																						this.jLabel6)
																				.addComponent(
																						this.text_3,
																						-2,
																						-1,
																						-2)
																				.addComponent(
																						this.btn_3))
																.addGap(18, 18,
																		18)
																.addGroup(
																		layout.createParallelGroup(
																				GroupLayout.Alignment.BASELINE)
																				.addComponent(
																						this.jLabel5)
																				.addComponent(
																						this.text_4,
																						-2,
																						-1,
																						-2)
																				.addComponent(
																						this.btn_4)))
												.addGroup(
														GroupLayout.Alignment.TRAILING,
														layout.createSequentialGroup()
																.addPreferredGap(
																		LayoutStyle.ComponentPlacement.RELATED)
																.addComponent(
																		this.jButton5,
																		-2, 50,
																		-2)))
								.addPreferredGap(
										LayoutStyle.ComponentPlacement.RELATED)
								.addComponent(this.jLabel4)
								.addContainerGap(29, 32767)));

		pack();
	}

	private void btn_1ActionPerformed(ActionEvent evt) {
		File f;
		if ((this.text_1.getText() != null)
				|| (!this.text_1.getText().trim().equals("")))
			f = new File(this.text_1.getText());
		else {
			f = new File(".");
		}
		JFileChooser jc = new JFileChooser(f);
		jc.setDialogTitle("选择文件");
		jc.setFileSelectionMode(1);
		if (0 == jc.showOpenDialog(getContentPane()))
			this.text_1.setText(jc.getSelectedFile().getPath());
	}

	private void btn_2ActionPerformed(ActionEvent evt) {
		File f;
		if ((this.text_2.getText() != null)
				|| (!this.text_2.getText().trim().equals("")))
			f = new File(this.text_2.getText());
		else {
			f = new File(".");
		}
		JFileChooser jc = new JFileChooser(f);
		jc.setDialogTitle("选择文件");
		jc.setFileSelectionMode(1);
		if (0 == jc.showOpenDialog(getContentPane()))
			this.text_2.setText(jc.getSelectedFile().getPath());
	}

	private void btn_3ActionPerformed(ActionEvent evt) {
		File f;
		if ((this.text_3.getText() != null)
				|| (!this.text_3.getText().trim().equals("")))
			f = new File(this.text_3.getText());
		else {
			f = new File(".");
		}
		JFileChooser jc = new JFileChooser(f);
		jc.setDialogTitle("选择文件");
		jc.setFileSelectionMode(1);
		if (0 == jc.showOpenDialog(getContentPane()))
			this.text_3.setText(jc.getSelectedFile().getPath());
	}

	private void btn_4ActionPerformed(ActionEvent evt) {
		File f;
		if ((this.text_4.getText() != null)
				|| (!this.text_4.getText().trim().equals("")))
			f = new File(this.text_4.getText());
		else {
			f = new File(".");
		}
		JFileChooser jc = new JFileChooser(f);
		jc.setDialogTitle("选择文件");
		jc.setFileSelectionMode(1);
		if (0 == jc.showOpenDialog(getContentPane()))
			this.text_4.setText(jc.getSelectedFile().getPath());
	}

	private void jButton3ActionPerformed(ActionEvent evt) {
		String inPath = this.text_1.getText();
		String outPath = this.text_2.getText();
		if ((inPath == null) || (inPath.trim().equals(""))) {
			JOptionPane.showMessageDialog(this, "输入路径不能为空");
			return;
		}
		if ((outPath == null) || (outPath.trim().equals(""))) {
			JOptionPane.showMessageDialog(this, "输出路径不能为空");
			return;
		}
		if (inPath.trim().equals(outPath.trim())) {
			JOptionPane.showMessageDialog(this, "输入输出路径不能相同");
			return;
		}
		Action.ExeLaout(inPath, outPath, this.jButton3, this);
	}

	private void jButton5ActionPerformed(ActionEvent evt) {
		String inPath = this.text_3.getText();
		String outPath = this.text_4.getText();
		if ((inPath == null) || (inPath.trim().equals(""))) {
			JOptionPane.showMessageDialog(this, "输入路径不能为空");
			return;
		}
		if ((outPath == null) || (outPath.trim().equals(""))) {
			JOptionPane.showMessageDialog(this, "输出路径不能为空");
			return;
		}
		if (inPath.trim().equals(outPath.trim())) {
			JOptionPane.showMessageDialog(this, "输入输出路径不能相同");
			return;
		}

		Action.ExeStyle(inPath, outPath, this.jButton5, this);
	}

	public static void main(String[] args) {
		try {
			for (UIManager.LookAndFeelInfo info : UIManager
					.getInstalledLookAndFeels())
				if ("Nimbus".equals(info.getName())) {
					UIManager.setLookAndFeel(info.getClassName());
					break;
				}
		} catch (ClassNotFoundException ex) {
			Logger.getLogger(NewJFrame.class.getName()).log(Level.SEVERE, null,
					ex);
		} catch (InstantiationException ex) {
			Logger.getLogger(NewJFrame.class.getName()).log(Level.SEVERE, null,
					ex);
		} catch (IllegalAccessException ex) {
			Logger.getLogger(NewJFrame.class.getName()).log(Level.SEVERE, null,
					ex);
		} catch (UnsupportedLookAndFeelException ex) {
			Logger.getLogger(NewJFrame.class.getName()).log(Level.SEVERE, null,
					ex);
		}

		EventQueue.invokeLater(new Runnable() {
			public void run() {
				new NewJFrame().setVisible(true);
			}
		});
	}
}