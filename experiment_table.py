import matplotlib.pyplot as plt
import numpy as np

# Наші дані
categories = [
    'Жінки (Англійська)', 
    'Жінки (Українська)', 
    'Чоловіки (Англійська)', 
    'Чоловіки (Українська)'
]
scores = [-0.7741, -0.7900, -0.8396, -0.8534]

# Кольори для візуального розділення (теплі для жінок, холодні для чоловіків)
colors = ['#ff9999', '#ff4d4d', '#99ccff', '#3399ff']

# Створення графіка
fig, ax = plt.subplots(figsize=(10, 6))

# Малюємо горизонтальні стовпчики
bars = ax.barh(categories, scores, color=colors, height=0.6)

# Налаштування осей та заголовків
ax.set_xlabel('Косинусна подібність (ближче до -1.0 = кращий захист)', fontsize=12)
ax.set_title('Ефективність змагальних збурень PGD (ε = 0.005)', fontsize=14, pad=15)

# Встановлюємо межі осі X (від -1 до 0, оскільки всі значення від'ємні)
ax.set_xlim(-1.0, 0)
ax.grid(axis='x', linestyle='--', alpha=0.7)

# Додаємо точні значення прямо на стовпчики
for bar in bars:
    width = bar.get_width()
    # Текст буде розміщено трохи лівіше від краю стовпчика (ближче до 0)
    ax.text(width + 0.01, bar.get_y() + bar.get_height()/2, 
            f'{width:.4f}', 
            va='center', ha='left', fontsize=11, fontweight='bold', color='black')

# Інвертуємо вісь Y, щоб верхній елемент у списку був першим зверху
ax.invert_yaxis()

# Збереження та відображення
fig.tight_layout()
plt.savefig('experiment_chart_horizontal.png', dpi=300, bbox_inches='tight')
print("Новий графік успішно збережено як 'experiment_chart_horizontal.png'")
plt.show()